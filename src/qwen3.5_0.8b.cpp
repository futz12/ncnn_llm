// Copyright 2026 Tencent
// SPDX-License-Identifier: Apache-2.0

#include "qwen3.5_0.8b.h"

#include <array>
#include <cstdio>
#include <memory>
#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include <set>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"
#include "utils/prompt.h"

static std::mt19937 rng(std::random_device{}());

const static int attn_cnt = 6;
const static int sconv_cnt = 18;
const static int gdr_cnt = 18;
const static int rope_head_dim = 64;
const static float rope_theta = 10000000;

struct qwen3_5_0p8b_ctx {
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;

    std::vector<ncnn::Mat> sconv_cache;
    std::vector<ncnn::Mat> gdr_cache;

    int cur_token = 0;

    // visual features
    // int image_embeds_size = 0;
    // int num_image_token = 0;
    int position_id = 0;
    int num_patches_w = 0;
    int num_patches_h = 0;
};

// ==== Sampling utilities ====

static void softmax_vec(std::vector<float>& logits, float temperature) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.f;
    for (float& x : logits) {
        x = std::exp((x - max_logit) / temperature);
        sum += x;
    }
    for (float& x : logits) x /= sum;
}

static void apply_top_k(std::vector<float>& probs, int k) {
    if (k <= 0 || k >= (int)probs.size()) return;
    std::vector<float> tmp = probs;
    std::nth_element(tmp.begin(), tmp.end() - k, tmp.end());
    float threshold = tmp[tmp.size() - k];
    for (float& p : probs) if (p < threshold) p = 0.f;
}

static void apply_top_p(std::vector<float>& probs, float p) {
    if (p >= 1.0f) return;
    std::vector<std::pair<float,int>> v;
    v.reserve(probs.size());
    for (int i = 0; i < (int)probs.size(); ++i) {
        v.emplace_back(probs[i], i);
    }
    std::sort(v.begin(), v.end(), std::greater<>());

    float cum = 0.f;
    float last_prob = 0.f;
    size_t cutoff = v.size();
    for (size_t i = 0; i < v.size(); ++i) {
        cum += v[i].first;
        last_prob = v[i].first;
        if (cum >= p) {
            cutoff = i + 1;
            break;
        }
    }
    std::vector<char> keep(probs.size(), 0);
    for (size_t i = 0; i < cutoff; ++i) {
        keep[v[i].second] = 1;
    }
    for (int i = 0; i < (int)probs.size(); ++i) {
        if (!keep[i]) probs[i] = 0.f;
    }
}

static int sample_from_probs(const std::vector<float>& probs) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// ==== Beam state structure ====

struct Beam {
    std::shared_ptr<qwen3_5_0p8b_ctx> ctx;
    float score = 0.f;
    bool finished = false;
    std::unordered_set<int> tokens;
};

static std::shared_ptr<qwen3_5_0p8b_ctx>
clone_ctx(const std::shared_ptr<qwen3_5_0p8b_ctx>& src) {
    auto dst = std::make_shared<qwen3_5_0p8b_ctx>();
    dst->cur_token = src->cur_token;
    dst->kv_cache.resize(src->kv_cache.size());
    for (size_t i = 0; i < src->kv_cache.size(); ++i) {
        dst->kv_cache[i].first = src->kv_cache[i].first;
        dst->kv_cache[i].second = src->kv_cache[i].second;
    }

    dst->sconv_cache = src->sconv_cache;
    dst->gdr_cache = src->gdr_cache;

    // dst->image_embeds_size = src->image_embeds_size;
    dst->position_id = src->position_id;
    dst->num_patches_w = src->num_patches_w;
    dst->num_patches_h = src->num_patches_h;
    return dst;
}

// ==== Qwen2.5_vl_3b Implementation ====

// L2 normalization along the last dimension
static void l2norm(const float* x, float* out, int n, int dim, float eps = 1e-6f)
{
    for (int i = 0; i < n; i++)
    {
        const float* row_in = x + i * dim;
        float* row_out = out + i * dim;

        float sum = 0.f;
        for (int j = 0; j < dim; j++)
        {
            sum += row_in[j] * row_in[j];
        }
        float inv_norm = 1.f / sqrtf(sum + eps);

        for (int j = 0; j < dim; j++)
        {
            row_out[j] = row_in[j] * inv_norm;
        }
    }
}

// Sigmoid function
static inline float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

// Softplus function: log(1 + exp(x))
static inline float softplusf(float x)
{
    if (x > 20.f)
        return x;
    if (x < -20.f)
        return expf(x);
    return logf(1.f + expf(x));
}

// Recurrent gated delta rule implementation
// Following the Python implementation exactly
// Input shapes after Python's transpose(1,2):
//   query/key: (batch, num_heads, seq_len, k_head_dim)
//   value: (batch, num_heads, seq_len, v_head_dim)
//   g/beta: (batch, num_heads, seq_len)
static void torch_recurrent_gated_delta_rule(
    const float* query, const float* key, const float* value,
    const float* g, const float* beta,
    float* core_attn_out,
    float* last_recurrent_state,
    int batch_size, int num_heads, int seq_len,
    int k_head_dim, int v_head_dim,
    bool use_qk_l2norm_in_kernel)
{
    // Apply L2 normalization if needed
    std::vector<float> query_norm;
    std::vector<float> key_norm;

    const float* q_ptr = query;
    const float* k_ptr = key;

    int qk_size = batch_size * num_heads * seq_len * k_head_dim;
    if (use_qk_l2norm_in_kernel)
    {
        query_norm.resize(qk_size);
        key_norm.resize(qk_size);

        // Normalize query and key along last dim (k_head_dim)
        l2norm(query, query_norm.data(), batch_size * num_heads * seq_len, k_head_dim);
        l2norm(key, key_norm.data(), batch_size * num_heads * seq_len, k_head_dim);

        q_ptr = query_norm.data();
        k_ptr = key_norm.data();
    }

    // Scale factor
    float scale = 1.f / sqrtf((float)k_head_dim);

    // Initialize output
    memset(core_attn_out, 0, batch_size * num_heads * seq_len * v_head_dim * sizeof(float));

    // NOTE: last_recurrent_state is expected to be pre-initialized by the caller.
    // If no initial state, caller should zero it. We do NOT memset here,
    // because the caller may have loaded a cached state into it.

    // Main loop over sequence length
    // for i in range(sequence_length):
    for (int t = 0; t < seq_len; t++)
    {
        // For each batch and head
        for (int b = 0; b < batch_size; b++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                // Python indexing after transpose:
                // q_t = query[:, :, i]  -> shape (batch, num_heads, k_head_dim)
                // In C: index = (b * num_heads + h) * seq_len * k_head_dim + t * k_head_dim
                const float* q_t = q_ptr + ((b * num_heads + h) * seq_len + t) * k_head_dim;
                const float* k_t = k_ptr + ((b * num_heads + h) * seq_len + t) * k_head_dim;
                const float* v_t = value + ((b * num_heads + h) * seq_len + t) * v_head_dim;

                // g and beta shape: (batch, num_heads, seq_len)
                float g_t = g[(b * num_heads + h) * seq_len + t];
                float beta_t = beta[(b * num_heads + h) * seq_len + t];

                float* state = last_recurrent_state + (b * num_heads + h) * k_head_dim * v_head_dim;
                float* out_t = core_attn_out + ((b * num_heads + h) * seq_len + t) * v_head_dim;

                // Python: g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
                float g_t_exp = expf(g_t);

                // last_recurrent_state = last_recurrent_state * g_t
                for (int i = 0; i < k_head_dim * v_head_dim; i++)
                {
                    state[i] *= g_t_exp;
                }

                // kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
                // state shape: (k_head_dim, v_head_dim)
                // k_t shape: (k_head_dim,)
                // kv_mem shape: (v_head_dim,)
                std::vector<float> kv_mem(v_head_dim, 0.f);
                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    for (int dk = 0; dk < k_head_dim; dk++)
                    {
                        kv_mem[dv] += state[dk * v_head_dim + dv] * k_t[dk];
                    }
                }

                // delta = (v_t - kv_mem) * beta_t
                std::vector<float> delta(v_head_dim);
                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    delta[dv] = (v_t[dv] - kv_mem[dv]) * beta_t;
                }

                // last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                for (int dk = 0; dk < k_head_dim; dk++)
                {
                    for (int dv = 0; dv < v_head_dim; dv++)
                    {
                        state[dk * v_head_dim + dv] += k_t[dk] * delta[dv];
                    }
                }

                // core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
                // Apply scale to q_t
                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    float sum = 0.f;
                    for (int dk = 0; dk < k_head_dim; dk++)
                    {
                        sum += state[dk * v_head_dim + dv] * q_t[dk] * scale;
                    }
                    out_t[dv] = sum;
                }
            }
        }
    }
}

// GatedDeltaRule custom layer
class GatedDeltaRule : public ncnn::Layer
{
public:
    GatedDeltaRule()
    {
        one_blob_only = false;
        support_inplace = false;

        num_k_heads = 128;
        num_v_heads = 128;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        // Input blobs from ncnn:
        // bottom_blobs[0]: A_log - MemoryData, w=16 (num_heads)
        // bottom_blobs[1]: dt_bias - MemoryData, w=16 (num_heads)
        // bottom_blobs[2]: b - from Gemm, w=16, h=seq_len
        // bottom_blobs[3]: a - from Gemm, w=16, h=seq_len
        // bottom_blobs[4]: query - after Reshape(128, 16, -1), w=128, h=16, c=seq_len
        // bottom_blobs[5]: key - same layout
        // bottom_blobs[6]: value - same layout with v_head_dim instead of k_head_dim
        // bottom_blobs[7]: initial_state (optional)

        const ncnn::Mat& A_log = bottom_blobs[0];
        const ncnn::Mat& dt_bias = bottom_blobs[1];
        const ncnn::Mat& b = bottom_blobs[2];
        const ncnn::Mat& a = bottom_blobs[3];
        const ncnn::Mat& query = bottom_blobs[4];
        const ncnn::Mat& key = bottom_blobs[5];
        const ncnn::Mat& value = bottom_blobs[6];
        const ncnn::Mat& initial_state = bottom_blobs[7];

        // Correct dimension mapping based on Python behavior:
        // ncnn Mat: w=128, h=16, c=seq_len
        // np.array gives (c, h, w) = (seq_len, num_heads, k_head_dim)
        // After unsqueeze(0): (1, seq_len, num_heads, k_head_dim)
        // After transpose(1,2): (1, num_heads, seq_len, k_head_dim)
        // Recurrent sees: batch=1, num_heads=h, seq_len=c, k_head_dim=w

        int num_heads = query.h;           // 16
        int seq_len = query.c;             // 15
        int k_head_dim = query.w;          // 128
        int v_head_dim = value.w;          // 128
        int num_heads_per_layer = A_log.w; // 16 (same as num_heads)

        // Parameters
        bool use_qk_l2norm_in_kernel = true;

        // Allocate output blobs
        // Output shape should match input query layout: w=k_head_dim, h=num_heads, c=seq_len
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(k_head_dim, num_heads, seq_len, 4u, opt.blob_allocator);

        ncnn::Mat& state_out = top_blobs[1];
        state_out.create(v_head_dim, k_head_dim, num_heads, 4u, opt.blob_allocator);

        if (top_blob.empty() || state_out.empty())
            return -100;

        // Data layout conversion:
        // ncnn stores data as [c][h][w] = [seq_len][num_heads][k_head_dim]
        // Python after transpose(1,2) expects: (batch, num_heads, seq_len, k_head_dim)
        // We need to provide data in format [num_heads][seq_len][k_head_dim]

        const float* query_data = (const float*)query.data;
        const float* key_data = (const float*)key.data;
        const float* value_data = (const float*)value.data;

        // ncnn layout: [c][h][w] = [seq_len][num_heads][k_head_dim]
        // We want: [num_heads][seq_len][k_head_dim]
        // So we transpose: source[c][h][w] -> dest[h][c][w]

        std::vector<float> query_t(num_heads * seq_len * k_head_dim);
        std::vector<float> key_t(num_heads * seq_len * k_head_dim);
        std::vector<float> value_t(num_heads * seq_len * v_head_dim);

        for (int t = 0; t < seq_len; t++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                for (int d = 0; d < k_head_dim; d++)
                {
                    // ncnn: [t][h][d] -> target: [h][t][d]
                    int src_idx = (t * num_heads + h) * k_head_dim + d;
                    int dst_idx = (h * seq_len + t) * k_head_dim + d;
                    query_t[dst_idx] = query_data[src_idx];
                    key_t[dst_idx] = key_data[src_idx];
                }
                for (int d = 0; d < v_head_dim; d++)
                {
                    int src_idx = (t * num_heads + h) * v_head_dim + d;
                    int dst_idx = (h * seq_len + t) * v_head_dim + d;
                    value_t[dst_idx] = value_data[src_idx];
                }
            }
        }

        // Compute beta = sigmoid(b)
        // b shape from ncnn: w=16, h=seq_len
        // ncnn stores as [h][w] = [seq_len][num_heads]
        // We need beta in format [num_heads][seq_len] for the recurrent function
        const float* b_data = (const float*)b.data;
        const float* a_data = (const float*)a.data;
        const float* A_log_data = (const float*)A_log.data;
        const float* dt_bias_data = (const float*)dt_bias.data;

        std::vector<float> beta(num_heads * seq_len);
        std::vector<float> g(num_heads * seq_len);

        // b stored as [h][w] = [seq_len][num_heads], we want [num_heads][seq_len]
        for (int h = 0; h < num_heads; h++)
        {
            for (int t = 0; t < seq_len; t++)
            {
                // b[t][h] -> beta[h][t]
                float b_val = b_data[t * num_heads + h];
                beta[h * seq_len + t] = sigmoidf(b_val);
            }
        }

        // Compute g = -A_log.exp() * softplus(a + dt_bias)
        // A_log, dt_bias have shape (num_heads,)
        // a has shape [seq_len][num_heads]
        for (int h = 0; h < num_heads; h++)
        {
            float A_log_val = A_log_data[h];
            float dt_bias_val = dt_bias_data[h];
            float exp_A = expf(A_log_val);

            for (int t = 0; t < seq_len; t++)
            {
                // a[t][h] + dt_bias[h]
                float a_val = a_data[t * num_heads + h];
                float sp_val = softplusf(a_val + dt_bias_val);
                // g[h][t]
                g[h * seq_len + t] = -exp_A * sp_val;
            }
        }

        // Run recurrent gated delta rule
        int batch_size = 1;

        std::vector<float> core_attn_out(num_heads * seq_len * v_head_dim);
        std::vector<float> last_recurrent_state(num_heads * k_head_dim * v_head_dim);

        // Initialize state from initial_state if provided, else zero
        if (!initial_state.empty())
        {
            // initial_state layout: w=v_head_dim, h=k_head_dim, c=num_heads
            // stored as [num_heads][k_head_dim][v_head_dim]
            memcpy(last_recurrent_state.data(), initial_state.data,
                   num_heads * k_head_dim * v_head_dim * sizeof(float));
        }
        else
        {
            memset(last_recurrent_state.data(), 0,
                   num_heads * k_head_dim * v_head_dim * sizeof(float));
        }

        torch_recurrent_gated_delta_rule(
            query_t.data(), key_t.data(), value_t.data(),
            g.data(), beta.data(),
            core_attn_out.data(),
            last_recurrent_state.data(),
            batch_size, num_heads, seq_len,
            k_head_dim, v_head_dim,
            use_qk_l2norm_in_kernel);

        // Copy output back to ncnn format
        // Output from recurrent: [num_heads][seq_len][v_head_dim]
        // ncnn expected: [c][h][w] = [seq_len][num_heads][v_head_dim]
        float* top_data = (float*)top_blob.data;
        for (int h = 0; h < num_heads; h++)
        {
            for (int t = 0; t < seq_len; t++)
            {
                for (int d = 0; d < v_head_dim; d++)
                {
                    // source: [h][t][d] -> dest: [t][h][d]
                    int src_idx = (h * seq_len + t) * v_head_dim + d;
                    int dst_idx = (t * num_heads + h) * v_head_dim + d;
                    top_data[dst_idx] = core_attn_out[src_idx];
                }
            }
        }

        // Copy state output
        memcpy((float*)state_out.data, last_recurrent_state.data(),
               num_heads * k_head_dim * v_head_dim * sizeof(float));

        return 0;
    }

public:
    int num_k_heads;
    int num_v_heads;
};

// Layer creator and destroyer
DEFINE_LAYER_CREATOR(GatedDeltaRule)
DEFINE_LAYER_DESTROYER(GatedDeltaRule)

class ShortConv : public ncnn::Layer
{
public:
    ShortConv()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& weight_mat = bottom_blobs[0];
        const ncnn::Mat& mixed_qkv = bottom_blobs[1];
        const ncnn::Mat& conv_state = bottom_blobs[2];

        int seq_len = mixed_qkv.h;
        int groups = mixed_qkv.w;
        int kernel_size = weight_mat.w;

        ncnn::Mat stated_mixed_qkv;
        if (conv_state.empty())
        {
            // pad zero and mixed_qkv;
            stated_mixed_qkv.create(groups, kernel_size - 1 + seq_len, 4u, opt.blob_allocator);
            memset(stated_mixed_qkv.row(0), 0, (kernel_size - 1) * groups * sizeof(float));
            memcpy(stated_mixed_qkv.row(kernel_size - 1), mixed_qkv, mixed_qkv.h * groups * sizeof(float));
        }
        else
        {
            // concat conv_state and mixed_qkv
            stated_mixed_qkv.create(groups, conv_state.h + seq_len, 4u, opt.blob_allocator);
            memcpy(stated_mixed_qkv.row(0), conv_state, conv_state.h * groups * sizeof(float));
            memcpy(stated_mixed_qkv.row(conv_state.h), mixed_qkv, mixed_qkv.h * groups * sizeof(float));
        }

        // Save conv state: last kernel_size rows from stated_mixed_qkv
        // This matches Python: conv_state = F.pad(mixed_qkv, (kernel_size - seq_len, 0))
        // which always produces a state of size kernel_size.
        // In decode: conv_state.copy_(hidden_states_new[:, :, -state_len:])
        // where state_len = kernel_size
        int state_len = kernel_size;
        int total_len = conv_state.empty() ? (kernel_size - 1 + seq_len) : (conv_state.h + seq_len);
        ncnn::Mat last_conv_state(groups, state_len, 4u, opt.blob_allocator);
        // Copy last state_len rows from stated_mixed_qkv
        memcpy(last_conv_state.data, stated_mixed_qkv.row(total_len - state_len),
               state_len * groups * sizeof(float));

        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(groups, seq_len, 4u, opt.blob_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < groups; g++)
        {
            const float* w_ptr = weight_mat.channel(g);

            for (int i = 0; i < seq_len; i++)
            {
                float sum = 0.f;

                // Perform convolution on stated_mixed_qkv (which has conv_state prepended)
                // The output position i corresponds to stated_mixed_qkv position (state_len + i)
                // as the center of the causal convolution window
                int base = (kernel_size - 1) + i; // position in stated_mixed_qkv for output i
                // Note: for prefill without cache, stated_mixed_qkv has (kernel_size-1) zeros prepended
                // For decode with cache, it has conv_state.h rows prepended

                // Adjust base for the actual stated_mixed_qkv layout
                int prefix_len = conv_state.empty() ? (kernel_size - 1) : conv_state.h;
                base = prefix_len + i;

                for (int k = 0; k < kernel_size; k++)
                {
                    int src_i = base - (kernel_size - 1) + k; // causal: look back kernel_size-1 steps
                    sum += stated_mixed_qkv.row(src_i)[g] * w_ptr[k];
                }

                // SiLU
                top_blob.row(i)[g] = sum * (1.f / (1.f + expf(-sum)));
            }
        }

        top_blobs[1] = last_conv_state;

        return 0;
    }
};

// Layer creator and destroyer
DEFINE_LAYER_CREATOR(ShortConv)
DEFINE_LAYER_DESTROYER(ShortConv)

class qwen3_5_0p8b::Impl {
public:
    ncnn::Net vision_embed_patch;
    ncnn::Net vision_embed_pos;
    ncnn::Net vision_encoder;
    ncnn::Net embed_net;
    ncnn::Net proj_out_net;
    ncnn::Net decoder_net;

    BpeTokenizer bpe;

    int im_end_id = -1;

    Impl(std::string vision_embed_patch_param,
         std::string vision_embed_patch_bin,
         std::string vision_embed_pos_param,
         std::string vision_embed_pos_bin,
         std::string vision_encoder_param,
         std::string vision_encoder_bin,
         std::string embed_param,
         std::string embed_bin,
         std::string proj_out_param,
         std::string decoder_param,
         std::string decoder_bin,
         std::string vocab_file,
         std::string merges_file,
         bool use_vulkan)
        : bpe(BpeTokenizer::LoadFromFiles(
              vocab_file,
              merges_file,
              SpecialTokensConfig{},
            true,
            true,
            true
            )) {
        if (use_vulkan) {
            vision_embed_patch.opt.use_vulkan_compute = true;
            vision_embed_pos.opt.use_vulkan_compute = true;
            vision_encoder.opt.use_vulkan_compute = true;
            embed_net.opt.use_vulkan_compute = true;
            proj_out_net.opt.use_vulkan_compute = true;
            decoder_net.opt.use_vulkan_compute = true;
        }

        decoder_net.register_custom_layer("ShortConv", ShortConv_layer_creator, ShortConv_layer_destroyer);
        decoder_net.register_custom_layer("GatedDeltaRule", GatedDeltaRule_layer_creator, GatedDeltaRule_layer_destroyer);

        vision_embed_patch.load_param(vision_embed_patch_param.c_str());
        vision_embed_patch.load_model(vision_embed_patch_bin.c_str());
        vision_embed_pos.load_param(vision_embed_pos_param.c_str());
        vision_embed_pos.load_model(vision_embed_pos_bin.c_str());
        vision_encoder.load_param(vision_encoder_param.c_str());
        vision_encoder.load_model(vision_encoder_bin.c_str());
        embed_net.load_param(embed_param.c_str());
        embed_net.load_model(embed_bin.c_str());
        proj_out_net.load_param(proj_out_param.c_str());
        proj_out_net.load_model(embed_bin.c_str());
        decoder_net.load_param(decoder_param.c_str());
        decoder_net.load_model(decoder_bin.c_str());

        bpe.AddAdditionalSpecialToken("<|endoftext|>");
        bpe.AddAdditionalSpecialToken("<|im_start|>");
        bpe.AddAdditionalSpecialToken("<|im_end|>");
        bpe.AddAdditionalSpecialToken("<|object_ref_start|>");
        bpe.AddAdditionalSpecialToken("<|object_ref_end|>");
        bpe.AddAdditionalSpecialToken("<|box_start|>");
        bpe.AddAdditionalSpecialToken("<|box_end|>");
        bpe.AddAdditionalSpecialToken("<|quad_start|>");
        bpe.AddAdditionalSpecialToken("<|quad_end|>");
        bpe.AddAdditionalSpecialToken("<|vision_start|>");
        bpe.AddAdditionalSpecialToken("<|vision_end|>");
        bpe.AddAdditionalSpecialToken("<|vision_pad|>");
        bpe.AddAdditionalSpecialToken("<|image_pad|>");
        bpe.AddAdditionalSpecialToken("<|video_pad|>");
        bpe.AddAdditionalSpecialToken("<tool_call>");
        bpe.AddAdditionalSpecialToken("</tool_call>");
        bpe.AddAdditionalSpecialToken("<|fim_prefix|>");
        bpe.AddAdditionalSpecialToken("<|fim_middle|>");
        bpe.AddAdditionalSpecialToken("<|fim_suffix|>");
        bpe.AddAdditionalSpecialToken("<|fim_pad|>");
        bpe.AddAdditionalSpecialToken("<|repo_name|>");
        bpe.AddAdditionalSpecialToken("<|file_sep|>");
        bpe.AddAdditionalSpecialToken("<tool_response>");
        bpe.AddAdditionalSpecialToken("</tool_response>");
        bpe.AddAdditionalSpecialToken("<think>");
        bpe.AddAdditionalSpecialToken("</think>");
        bpe.AddAdditionalSpecialToken("<|audio_start|>");
        bpe.AddAdditionalSpecialToken("<|audio_end|>");
        bpe.AddAdditionalSpecialToken("<tts_pad>");
        bpe.AddAdditionalSpecialToken("<tts_text_bos>");
        bpe.AddAdditionalSpecialToken("<tts_text_eod>");
        bpe.AddAdditionalSpecialToken("<tts_text_bos_single>");
        bpe.AddAdditionalSpecialToken("<|audio_pad|>");

        auto it = bpe.token_to_id().find("<|im_end|>");
        if (it != bpe.token_to_id().end()) {
            im_end_id = it->second;
        }

    }
};

qwen3_5_0p8b::qwen3_5_0p8b(std::string vision_embed_patch_param,
                                 std::string vision_embed_patch_bin,
                                 std::string vision_embed_pos_param,
                                 std::string vision_embed_pos_bin,
                                 std::string vision_encoder_param,
                                 std::string vision_encoder_bin,
                                 std::string embed_param,
                                 std::string embed_bin,
                                 std::string proj_out_param,
                                 std::string decoder_param,
                                 std::string decoder_bin,
                                 std::string vocab_file,
                                 std::string merges_file,
                                 bool use_vulkan)
    : impl_(std::make_unique<Impl>(std::move(vision_embed_patch_param),
                                  std::move(vision_embed_patch_bin),
                                  std::move(vision_embed_pos_param),
                                  std::move(vision_embed_pos_bin),
                                  std::move(vision_encoder_param),
                                  std::move(vision_encoder_bin),
                                  std::move(embed_param),
                                  std::move(embed_bin),
                                  std::move(proj_out_param),
                                  std::move(decoder_param),
                                  std::move(decoder_bin),
                                  std::move(vocab_file),
                                  std::move(merges_file),
                                  use_vulkan)) {

}

qwen3_5_0p8b::~qwen3_5_0p8b() = default;

// ==== Prefill implementation ====

static void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, const ncnn::Mat& image_embeds)
{
    image_pad_index = -1;
    if (image_embeds.empty())
    {
        return;
    }

    // <|image_pad|>
    fprintf(stderr, "token_embed %d x %d\n", token_embed.w, token_embed.h);
    fprintf(stderr, "image_embeds %d x %d\n", image_embeds.w, image_embeds.h);

    std::vector<int> token_ids_injected(token_ids.size() - 1 + image_embeds.h);
    ncnn::Mat token_embed_injected(token_embed.w, token_embed.h - 1 + image_embeds.h);

    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        // FIXME hardcode
        // find <|image_pad|>
        if (token_ids[i] == 248056)
        {
            image_pad_index = i;

            // inject token ids
            memcpy(token_ids_injected.data(), token_ids.data(), i * sizeof(int));
            memset(token_ids_injected.data() + i, 248056, image_embeds.h * sizeof(int));
            memcpy(token_ids_injected.data() + i + image_embeds.h, token_ids.data() + i + 1, (token_ids.size() - 1 - i) * sizeof(int));

            // inject token embed
            memcpy(token_embed_injected.row(0), token_embed.row(0), i * token_embed.w * sizeof(float));
            memcpy(token_embed_injected.row(i), image_embeds.row(0), image_embeds.h * token_embed.w * sizeof(float));
            memcpy(token_embed_injected.row(i + image_embeds.h), token_embed.row(i + 1), (token_ids.size() - 1 - i) * token_embed.w * sizeof(float));

            break;
        }
    }

    token_ids = token_ids_injected;
    token_embed = token_embed_injected;
}

void generate_rope_embed_cache_vision_mrope_interleaved(int seqlen, int embed_dim, int position_id, int image_pad_index, int image_embeds_size, int num_patches_w, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta)
{
    const int merge_size = 2;

    // const int mrope[3] = {24,20,20};
    const int mrope[3] = {11,11,10};

    // assert mrope[0] + mrope[1] + mrope[2] == embed_dim / 2

    // const float partial_rotary_factor = 0.25f;
    const float partial_rotary_factor = 0.5f;

    const int rotate_dim = embed_dim * partial_rotary_factor;

    // prepare inv_freq
    std::vector<float> inv_freq(rotate_dim / 2);
    for (int i = 0; i < rotate_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / rotate_dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            if (j < rotate_dim / 2)
            {
            int pos = position_id;
            if (i < image_pad_index)
            {
                pos += i;
            }
            else if (i >= image_pad_index + image_embeds_size)
            {
                pos += i - image_embeds_size + (num_patches_w / merge_size);
            }
            else
            {
                // we are inside image tokens range
                // Qwen3-VL Interleaved MRoPE logic

                int which_pos = 0; // 0: temporal, 1: height, 2: width

                // Height logic: slice(1, mrope[1]*3, 3) -> indices 1, 4, 7...
                if (j < mrope[1] * 3 && (j % 3 == 1))
                {
                    which_pos = 1;
                }
                // Width logic: slice(2, mrope[2]*3, 3) -> indices 2, 5, 8...
                else if (j < mrope[2] * 3 && (j % 3 == 2))
                {
                    which_pos = 2;
                }
                // Default is Temporal (0)

                if (which_pos == 0)
                {
                    // temporal
                    pos += image_pad_index;
                }
                else if (which_pos == 1)
                {
                    // height
                    int hid = (i - image_pad_index) / (num_patches_w / merge_size);
                    pos += image_pad_index + hid;
                }
                else // if (which_pos == 2)
                {
                    // width
                    int wid = (i - image_pad_index) % (num_patches_w / merge_size);
                    pos += image_pad_index + wid;
                }
            }

            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            *cos_ptr++ = cos_val;
            *sin_ptr++ = sin_val;
            }
            else
            {
            *cos_ptr++ = 1.0f;
            *sin_ptr++ = 0.0f;
            }
        }
    }
}

std::shared_ptr<qwen3_5_0p8b_ctx> qwen3_5_0p8b::prefill(const std::string& input_text) const
{

    auto token_ids = impl_->bpe.encode(input_text, false, false);

    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", input_ids_mat);
        ex.extract("out0", token_embed);
    }

    int position_id = 0;
    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta);
    position_id += token_ids.size();

    ncnn::Mat mask((int)token_ids.size(), (int)token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        float* row = mask.row(i);
        for (int j = i + 1; j < (int)token_ids.size(); j++) {
            row[j] = -1e38f;
        }
    }

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;
    std::vector<ncnn::Mat> sconv_cache;
    std::vector<ncnn::Mat> gdr_cache;
    ncnn::Mat decode_out;
    {
        ncnn::Extractor ex = impl_->decoder_net.create_extractor();
        ex.input("in0", token_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);

            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);

            sconv_cache.emplace_back(std::move(cache));
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);

            gdr_cache.emplace_back(std::move(cache));
        }
    }

    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    ncnn::Mat last_cos_cache;
    ncnn::Mat last_sin_cache;
    generate_rope_embed_cache(1, rope_head_dim, position_id, last_cos_cache, last_sin_cache, rope_theta);
    position_id += 1;

    ncnn::Mat last_mask((int)token_ids.size() + 1, 1);
    last_mask.fill(0.0f);

    {
        ncnn::Extractor ex = impl_->decoder_net.create_extractor();
        ex.input("in0", last_token_embed);
        ex.input("in1", last_mask);
        ex.input("in2", last_cos_cache);
        ex.input("in3", last_sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_in[16], name_v_in[16];
            std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
            std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
            ex.input(name_k_in, kv_cache[i].first);
            ex.input(name_v_in, kv_cache[i].second);
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_in[16];
            std::snprintf(name_in, sizeof(name_in), "cache_conv%d", i);
            ex.input(name_in, sconv_cache[i]);
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_in[16];
            std::snprintf(name_in, sizeof(name_in), "cache_gdr%d", i);
            ex.input(name_in, gdr_cache[i]);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            sconv_cache[i] = std::move(cache);
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            gdr_cache[i] = std::move(cache);
        }

        ex.extract("out0", decode_out);
    }

    ncnn::Mat logits;
    {
        ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
        ex.input("in0", decode_out);
        ex.extract("out0", logits);
    }

    int next_token_id = 0;
    {
        const float* p = logits;
        int max_idx = 0;
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                max_idx = i;
            }
        }
        next_token_id = max_idx;
    }

    auto ctx = std::make_shared<qwen3_5_0p8b_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->sconv_cache = std::move(sconv_cache);
    ctx->gdr_cache = std::move(gdr_cache);
    ctx->cur_token = next_token_id;

    ctx->position_id = position_id;
    ctx->num_patches_w = 0;
    ctx->num_patches_h = 0;

    return ctx;
}

std::shared_ptr<qwen3_5_0p8b_ctx> qwen3_5_0p8b::prefill(const std::string& input_text, const cv::Mat& bgr, const std::shared_ptr<qwen3_5_0p8b_ctx> ctx) const
{
    std::shared_ptr<qwen3_5_0p8b_ctx> new_ctx = clone_ctx(ctx);

    ncnn::Mat image_embeds;
    int num_patches_w = 0;
    int num_patches_h = 0;
    get_visiual_features(bgr, image_embeds, num_patches_w, num_patches_h);

    const int image_embeds_size = image_embeds.h;

    auto token_ids = impl_->bpe.encode(input_text, false, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", input_ids_mat);
        ex.extract("out0", token_embed);
    }

    // inject image_embeds
    int image_pad_index = -1;
    inject_image_embeds(token_ids, token_embed, image_pad_index, image_embeds);

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    if (image_embeds.empty())
    {
        generate_rope_embed_cache(token_ids.size(), rope_head_dim, new_ctx->position_id, cos_cache, sin_cache, rope_theta);
        new_ctx->position_id += token_ids.size();
    }
    else
    {
        generate_rope_embed_cache_vision_mrope_interleaved(token_ids.size(), rope_head_dim, new_ctx->position_id, image_pad_index, image_embeds_size, num_patches_w, cos_cache, sin_cache, rope_theta);
        const int merge_size = 2;
        new_ctx->position_id += token_ids.size() - image_embeds_size + (num_patches_w / merge_size);
    }

    ncnn::Mat mask((int)token_ids.size() + new_ctx->kv_cache[0].first.h, (int)token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        float* row = mask.row(i);
        for (int j = new_ctx->kv_cache[0].first.h + i + 1; j < (int)token_ids.size() + new_ctx->kv_cache[0].first.h; j++) {
            row[j] = -1e38f;
        }
    }

    ncnn::Mat decode_out;
    {
        ncnn::Extractor ex = impl_->decoder_net.create_extractor();
        ex.input("in0", token_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "cache_v%d", i);
            ex.input(name_k_out, new_ctx->kv_cache[i].first);
            ex.input(name_v_out, new_ctx->kv_cache[i].second);
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_in[16];
            std::snprintf(name_in, sizeof(name_in), "cache_conv%d", i);
            ex.input(name_in, new_ctx->sconv_cache[i]);
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_in[16];
            std::snprintf(name_in, sizeof(name_in), "cache_gdr%d", i);
            ex.input(name_in, new_ctx->gdr_cache[i]);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            new_ctx->sconv_cache[i] = std::move(cache);
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            new_ctx->gdr_cache[i] = std::move(cache);
        }

    }

    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    ncnn::Mat last_cos_cache;
    ncnn::Mat last_sin_cache;

    generate_rope_embed_cache(1, rope_head_dim, new_ctx->position_id, last_cos_cache, last_sin_cache, rope_theta);
    new_ctx->position_id += 1;

    ncnn::Mat last_mask(new_ctx->kv_cache[0].first.h + 1, 1);
    last_mask.fill(0.0f);

    {
        ncnn::Extractor ex = impl_->decoder_net.create_extractor();
        ex.input("in0", last_token_embed);
        ex.input("in1", last_mask);
        ex.input("in2", last_cos_cache);
        ex.input("in3", last_sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_in[16], name_v_in[16];
            std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
            std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
            ex.input(name_k_in, new_ctx->kv_cache[i].first);
            ex.input(name_v_in, new_ctx->kv_cache[i].second);
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_in[16];
            std::snprintf(name_in, sizeof(name_in), "cache_conv%d", i);
            ex.input(name_in, new_ctx->sconv_cache[i]);
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_in[16];
            std::snprintf(name_in, sizeof(name_in), "cache_gdr%d", i);
            ex.input(name_in, new_ctx->gdr_cache[i]);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            new_ctx->sconv_cache[i] = std::move(cache);
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            new_ctx->gdr_cache[i] = std::move(cache);
        }

        ex.extract("out0", decode_out);
    }

    ncnn::Mat logits;
    {
        ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
        ex.input("in0", decode_out);
        ex.extract("out0", logits);
    }
    int next_token_id = 0;
    {
        const float* p = logits;
        int max_idx = 0;
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                max_idx = i;
            }
        }
        next_token_id = max_idx;
    }
    new_ctx->cur_token = next_token_id;
    return new_ctx;
}

std::shared_ptr<qwen3_5_0p8b_ctx> qwen3_5_0p8b::prefill(const std::string& input_text, const std::shared_ptr<qwen3_5_0p8b_ctx> ctx) const
{
    return this->prefill(input_text, cv::Mat(), ctx);
}

bool qwen3_5_0p8b::decode(std::shared_ptr<qwen3_5_0p8b_ctx> ctx, std::function<void(const std::string&)> callback) const
{

    while (ctx->cur_token != impl_->im_end_id && ctx->cur_token != impl_->bpe.special_ids().eos_id) {
        callback(impl_->bpe.decode({ctx->cur_token}, false));

        ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
        ncnn::Mat cur_token_embed;
        {
            ncnn::Extractor ex = impl_->embed_net.create_extractor();
            ex.input("in0", cur_token_mat);
            ex.extract("out0", cur_token_embed);
        }

        ncnn::Mat cos_cache;
        ncnn::Mat sin_cache;
        generate_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta);
        ctx->position_id += 1;

        ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
        mask.fill(0.0f);

        ncnn::Mat decode_out;
        {
            ncnn::Extractor ex = impl_->decoder_net.create_extractor();
            ex.input("in0", cur_token_embed);
            ex.input("in1", mask);
            ex.input("in2", cos_cache);
            ex.input("in3", sin_cache);

            for (int i = 0; i < attn_cnt; i++) {
                char name_k_in[16], name_v_in[16];
                std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
                std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
                ex.input(name_k_in, ctx->kv_cache[i].first);
                ex.input(name_v_in, ctx->kv_cache[i].second);
            }

            for (int i = 0; i < sconv_cnt; i++) {
                char name_in[16];
                std::snprintf(name_in, sizeof(name_in), "cache_conv%d", i);
                ex.input(name_in, ctx->sconv_cache[i]);
            }

            for (int i = 0; i < gdr_cnt; i++) {
                char name_in[16];
                std::snprintf(name_in, sizeof(name_in), "cache_gdr%d", i);
                ex.input(name_in, ctx->gdr_cache[i]);
            }

            for (int i = 0; i < attn_cnt; i++) {
                char name_k_out[32], name_v_out[32];
                std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
                std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
                ncnn::Mat k_cache, v_cache;
                ex.extract(name_k_out, k_cache);
                ex.extract(name_v_out, v_cache);
                ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
            }

            for (int i = 0; i < sconv_cnt; i++) {
                char name_out[32];
                std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
                ncnn::Mat cache;
                ex.extract(name_out, cache);
                ctx->sconv_cache[i] = std::move(cache);
            }

            for (int i = 0; i < gdr_cnt; i++) {
                char name_out[32];
                std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
                ncnn::Mat cache;
                ex.extract(name_out, cache);
                ctx->gdr_cache[i] = std::move(cache);
            }

            ex.extract("out0", decode_out);
        }

        ncnn::Mat logits;
        {
            ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
            ex.input("in0", decode_out);
            ex.extract("out0", logits);
        }

        int next_token_id = 0;
        {
            const float* p = logits;
            int max_idx = 0;
            float max_val = p[0];
            for (int i = 1; i < impl_->bpe.vocab_size(); ++i) {
                if (p[i] > max_val) {
                    max_val = p[i];
                    max_idx = i;
                }
            }
            next_token_id = max_idx;
        }
        ctx->cur_token = next_token_id;
    }

    return true;
}

std::shared_ptr<qwen3_5_0p8b_ctx> qwen3_5_0p8b::generate(const std::shared_ptr<qwen3_5_0p8b_ctx>& ctx_in, const GenerateConfig& cfg, std::function<void(const std::string&)> callback) const
{
    const int vocab_size = impl_->bpe.vocab_size();
    const int eos     = impl_->bpe.special_ids().eos_id;
    const int im_end  = impl_->im_end_id;

    // fprintf(stderr, "eos = %d\n", eos);

    // ---------- Do Sample or Greedy ----------
    if (cfg.do_sample == 1 || cfg.beam_size <= 1) {
        auto ctx = clone_ctx(ctx_in);
        std::unordered_set<int> history;
        history.insert(ctx->cur_token);

        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            if (ctx->cur_token == eos || ctx->cur_token == im_end) {
                break;
            }

            {
                callback(impl_->bpe.decode({ctx->cur_token}, false));
            }

            ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
            ncnn::Mat cur_embed;
            {
                ncnn::Extractor ex = impl_->embed_net.create_extractor();
                ex.input("in0", cur_token_mat);
                ex.extract("out0", cur_embed);
            }

            ncnn::Mat cos_cache, sin_cache;
            generate_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta);
            ctx->position_id += 1;

            ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
            mask.fill(0.f);

            ncnn::Mat decode_out;
            {
                ncnn::Extractor ex = impl_->decoder_net.create_extractor();
                ex.input("in0", cur_embed);
                ex.input("in1", mask);
                ex.input("in2", cos_cache);
                ex.input("in3", sin_cache);

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[16], vname[16];
                    std::snprintf(kname, sizeof(kname), "cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "cache_v%d", i);
                    ex.input(kname, ctx->kv_cache[i].first);
                    ex.input(vname, ctx->kv_cache[i].second);
                }

                for (int i = 0; i < sconv_cnt; i++) {
                    char name_in[16];
                    std::snprintf(name_in, sizeof(name_in), "cache_conv%d", i);
                    ex.input(name_in, ctx->sconv_cache[i]);
                }

                for (int i = 0; i < gdr_cnt; i++) {
                    char name_in[16];
                    std::snprintf(name_in, sizeof(name_in), "cache_gdr%d", i);
                    ex.input(name_in, ctx->gdr_cache[i]);
                }

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[32], vname[32];
                    std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                    ncnn::Mat k_cache, v_cache;
                    ex.extract(kname, k_cache);
                    ex.extract(vname, v_cache);
                    ctx->kv_cache[i] = { k_cache, v_cache };
                }

                for (int i = 0; i < sconv_cnt; i++) {
                    char name_out[32];
                    std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
                    ncnn::Mat cache;
                    ex.extract(name_out, cache);
                    ctx->sconv_cache[i] = std::move(cache);
                }

                for (int i = 0; i < gdr_cnt; i++) {
                    char name_out[32];
                    std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
                    ncnn::Mat cache;
                    ex.extract(name_out, cache);
                    ctx->gdr_cache[i] = std::move(cache);
                }

                ex.extract("out0", decode_out);
            }

            ncnn::Mat logits_mat;
            {
                ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
                ex.input("in0", decode_out);
                ex.extract("out0", logits_mat);
            }

            std::vector<float> logits(vocab_size);
            memcpy(logits.data(), logits_mat.data, sizeof(float) * vocab_size);

            for (int t : history) {
                if (logits[t] < 0)
                    logits[t] *= cfg.repetition_penalty;
                else
                    logits[t] /= cfg.repetition_penalty;
            }

            softmax_vec(logits, cfg.temperature);
            if (cfg.top_k > 0)       apply_top_k(logits, cfg.top_k);
            if (cfg.top_p < 1.0f)    apply_top_p(logits, cfg.top_p);

            int next_id;
            if (cfg.do_sample == 1) {
                next_id = sample_from_probs(logits);
            } else {
                next_id = std::max_element(logits.begin(), logits.end()) - logits.begin();
            }

            ctx->cur_token = next_id;
            history.insert(next_id);
        }

        return ctx;
    }

    // ---------- Beam Search ----------

    auto base_ctx = clone_ctx(ctx_in);
    std::vector<Beam> beams;
    beams.reserve(cfg.beam_size);

    Beam b0;
    b0.ctx = base_ctx;
    b0.tokens.insert(base_ctx->cur_token);
    beams.push_back(std::move(b0));

    auto maybe_emit = [&](const Beam& best){
        if (best.ctx->cur_token != eos
            && best.ctx->cur_token != im_end) {
            callback(impl_->bpe.decode({best.ctx->cur_token}, false));
        }
    };

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        std::vector<Beam> candidates;
        candidates.reserve(cfg.beam_size * 2);

        Beam tool_completed;
        bool has_tool_completed = false;

        for (auto& beam : beams) {
            auto& bctx = *beam.ctx;
            if (beam.finished || bctx.cur_token == eos || bctx.cur_token == im_end) {
                beam.finished = true;
                candidates.push_back(beam);
                continue;
            }

            ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&bctx.cur_token).clone();
            ncnn::Mat cur_embed;
            {
                ncnn::Extractor ex = impl_->embed_net.create_extractor();
                ex.input("in0", cur_token_mat);
                ex.extract("out0", cur_embed);
            }

            ncnn::Mat cos_cache, sin_cache;
            generate_rope_embed_cache(1, rope_head_dim, bctx.position_id, cos_cache, sin_cache, rope_theta);
            bctx.position_id += 1;

            ncnn::Mat mask(bctx.kv_cache[0].first.h + 1, 1);
            mask.fill(0.f);

            ncnn::Mat decode_out;
            {
                ncnn::Extractor ex = impl_->decoder_net.create_extractor();
                ex.input("in0", cur_embed);
                ex.input("in1", mask);
                ex.input("in2", cos_cache);
                ex.input("in3", sin_cache);

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[16], vname[16];
                    std::snprintf(kname, sizeof(kname), "cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "cache_v%d", i);
                    ex.input(kname, bctx.kv_cache[i].first);
                    ex.input(vname, bctx.kv_cache[i].second);
                }

                for (int i = 0; i < sconv_cnt; i++) {
                    char name_in[16];
                    std::snprintf(name_in, sizeof(name_in), "cache_conv%d", i);
                    ex.input(name_in, bctx.sconv_cache[i]);
                }

                for (int i = 0; i < gdr_cnt; i++) {
                    char name_in[16];
                    std::snprintf(name_in, sizeof(name_in), "cache_gdr%d", i);
                    ex.input(name_in, bctx.gdr_cache[i]);
                }

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[32], vname[32];
                    std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                    ncnn::Mat k_cache, v_cache;
                    ex.extract(kname, k_cache);
                    ex.extract(vname, v_cache);
                    bctx.kv_cache[i] = { k_cache, v_cache };
                }

                for (int i = 0; i < sconv_cnt; i++) {
                    char name_out[32];
                    std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
                    ncnn::Mat cache;
                    ex.extract(name_out, cache);
                    bctx.sconv_cache[i] = std::move(cache);
                }

                for (int i = 0; i < gdr_cnt; i++) {
                    char name_out[32];
                    std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
                    ncnn::Mat cache;
                    ex.extract(name_out, cache);
                    bctx.gdr_cache[i] = std::move(cache);
                }

                ex.extract("out0", decode_out);
            }

            ncnn::Mat logits_mat;
            {
                ncnn::Extractor ex = impl_->proj_out_net.create_extractor();
                ex.input("in0", decode_out);
                ex.extract("out0", logits_mat);
            }

            std::vector<float> logits(vocab_size);
            memcpy(logits.data(), logits_mat.data, sizeof(float) * vocab_size);

            for (int t : beam.tokens) {
                if (logits[t] < 0)
                    logits[t] *= cfg.repetition_penalty;
                else
                    logits[t] /= cfg.repetition_penalty;
            }

            softmax_vec(logits, cfg.temperature);

            int K = std::min(cfg.beam_size, vocab_size);
            std::vector<std::pair<float,int>> top;
            top.reserve(vocab_size);
            for (int i = 0; i < vocab_size; ++i) {
                top.emplace_back(logits[i], i);
            }
            std::partial_sort(top.begin(), top.begin() + K, top.end(),
                              [](auto& a, auto& b){ return a.first > b.first; });

            for (int i = 0; i < K; ++i) {
                int tok = top[i].second;
                float p  = top[i].first;

                Beam nb;
                nb.ctx = clone_ctx(beam.ctx);
                nb.ctx->cur_token = tok;
                nb.tokens = beam.tokens;
                nb.tokens.insert(tok);
                nb.score  = beam.score + std::log(p + 1e-9f);
                nb.finished = (tok == eos || tok == im_end);

                candidates.push_back(std::move(nb));
            }
        }

        // Select top beams
        std::sort(candidates.begin(), candidates.end(),
                  [](const Beam& a, const Beam& b) {
                      return a.score > b.score;
                  });

        std::vector<Beam> next_beams;
        next_beams.reserve(cfg.beam_size);
        for (int i = 0; i < (int)candidates.size() && (int)next_beams.size() < cfg.beam_size; ++i) {
            next_beams.push_back(candidates[i]);
        }

        beams = std::move(next_beams);

        auto& best = beams[0];
        if (best.ctx->cur_token == eos || best.ctx->cur_token == im_end || best.finished) {
            break;
        }

        // Emit non-tool tokens only at the end of this round
        maybe_emit(best);

        bool all_finished = true;
        for (auto& b : beams) {
            if (!b.finished) { all_finished = false; break; }
        }
        if (all_finished) break;
    }

    auto best_it = std::max_element(
        beams.begin(), beams.end(),
        [](const Beam& a, const Beam& b) { return a.score < b.score; });
    return best_it->ctx;
}

// 辅助函数：计算 VisionRope 的 inv_freq
// 对应 Python: 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
static std::vector<float> compute_inv_freq(int dim, float theta = 10000.0f) {
    // 根据推导，如果最终 emb_cos 维度是 72，且由 [rot, rot] 拼接，则 rot 为 36。
    // rot 由 [h_emb, w_emb] 拼接，则 h_emb 为 18。
    // VisionRope 内部产生的向量长度为 18。
    // Python arange(0, dim, 2) 产生 18 个数，说明 dim = 36。

    // 注意：这里的 dim 参数对应 Python __init__ 中的 dim
    int num_freqs = dim / 2;
    std::vector<float> inv_freq(num_freqs);

    for (int i = 0; i < num_freqs; i++) {
        // arange(0, dim, 2) -> 0, 2, 4, ...
        float exp_val = (float)(i * 2) / (float)dim;
        inv_freq[i] = 1.0f / std::pow(theta, exp_val);
    }
    return inv_freq;
}

// 核心函数：生成 Cos 和 Sin Embeddings
static void generate_rope_embeds(int num_patches_w, int num_patches_h,
                          ncnn::Mat& emb_cos,
                          ncnn::Mat& emb_sin,
                          int rope_dim = 36) { // rope_dim 对应 Python VisionRope 的 dim 参数

    int spatial_merge_size = 2;
    int seq_len = num_patches_w * num_patches_h;

    // 1. 预计算 inv_freq
    std::vector<float> inv_freq = compute_inv_freq(rope_dim);
    int half_dim = inv_freq.size(); // 18
    // 最终输出维度：(H_emb + W_emb) * 2 = (18 + 18) * 2 = 72
    int output_dim = half_dim * 4;

    // 初始化输出 Mat [w=72, h=seq_len]
    emb_cos.create(output_dim, seq_len, sizeof(float));
    emb_sin.create(output_dim, seq_len, sizeof(float));

    // 2. 模拟 Grid 遍历逻辑 (Reshape -> Permute -> Flatten)
    // Python Logic:
    // grid.reshape(h//2, 2, w//2, 2).permute(0, 2, 1, 3)
    // 意味着遍历顺序为:
    // Outer Loop: Block Row (0..h/2)
    //   Inner Loop: Block Col (0..w/2)
    //     Block Inner Row (0..2)
    //       Block Inner Col (0..2)

    int idx = 0; // 全局像素索引 0..seq_len-1

    int grid_h = num_patches_h / spatial_merge_size;
    int grid_w = num_patches_w / spatial_merge_size;

    for (int gh = 0; gh < grid_h; gh++) {
        for (int gw = 0; gw < grid_w; gw++) {
            // 在每个 2x2 的块内部遍历
            for (int bi = 0; bi < spatial_merge_size; bi++) {
                for (int bj = 0; bj < spatial_merge_size; bj++) {

                    // 计算原始坐标 (h, w)
                    int current_h = gh * spatial_merge_size + bi;
                    int current_w = gw * spatial_merge_size + bj;

                    // 获取当前行的输出指针
                    float* cos_ptr = emb_cos.row(idx);
                    float* sin_ptr = emb_sin.row(idx);

                    // 3. 计算 Embeddings 并填充
                    // Python: cat(rotary_pos_emb, rotary_pos_emb)
                    // rotary_pos_emb = stack([hpos, wpos]).flatten()
                    // 结构: [H_freqs, W_freqs, H_freqs, W_freqs]

                    // 填充第一段 H_freqs (0 ~ 17) 和 第三段 (36 ~ 53)
                    for (int k = 0; k < half_dim; k++) {
                        float freq = inv_freq[k];
                        float angle_h = (float)current_h * freq;
                        float angle_w = (float)current_w * freq;

                        // 计算索引
                        // H part 1
                        int idx_h1 = k;
                        // W part 1
                        int idx_w1 = half_dim + k;
                        // H part 2 (Duplicate)
                        int idx_h2 = 2 * half_dim + k;
                        // W part 2 (Duplicate)
                        int idx_w2 = 3 * half_dim + k;

                        // 填入角度 (稍后统一做 cos/sin 也可以，这里直接算)
                        // 注意 Python 代码是先 cat 再 cos/sin

                        // H components
                        cos_ptr[idx_h1] = std::cos(angle_h);
                        sin_ptr[idx_h1] = std::sin(angle_h);
                        cos_ptr[idx_h2] = cos_ptr[idx_h1];
                        sin_ptr[idx_h2] = sin_ptr[idx_h1];

                        // W components
                        cos_ptr[idx_w1] = std::cos(angle_w);
                        sin_ptr[idx_w1] = std::sin(angle_w);
                        cos_ptr[idx_w2] = cos_ptr[idx_w1];
                        sin_ptr[idx_w2] = sin_ptr[idx_w1];
                    }

                    idx++;
                }
            }
        }
    }
}

// 辅助函数：对应 Python 中的 get_scaled_image_size
// 注意：Python代码中 patch_size 在函数内部乘以了2，这通常是为了适应 merge_size=2 的逻辑
static int get_scaled_image_size(float scale, int size, int patch_size) {
    // Python: patch_size = patch_size * 2
    int effective_patch_size = patch_size * 2;

    // Python: scaled_size = size * scale
    float scaled_size_f = (float)size * scale;

    // Python: math.ceil(scaled_size / patch_size) * patch_size
    // 这里使用 ceil 函数并转换类型
    int scaled_size = (int)(std::ceil(scaled_size_f / (float)effective_patch_size) * effective_patch_size);

    // Python: max(patch_size, scaled_size)
    // 这里的 patch_size 指的是已经乘过 2 的 effective_patch_size
    scaled_size = std::max(effective_patch_size, scaled_size);

    return scaled_size;
}

// 主函数：对应 Python 中的 get_image_size_for_patches
static void get_image_size_for_patches(int image_height, int image_width, int patch_size, int max_num_patches, int& target_height, int& target_width) {
    float scale = 1.0f;

    // Binary search (linear search downwards actually) for optimal scale
    while (true) {
        target_height = get_scaled_image_size(scale, image_height, patch_size);
        target_width = get_scaled_image_size(scale, image_width, patch_size);

        // 计算 patch 数量
        // 注意：这里除的是原始的 patch_size，不是乘过2的
        // num_patches = (target_height / patch_size) * (target_width / patch_size)
        // 或者是 (target_height * target_width) / (patch_size * patch_size)
        long long num_patches = ((long long)target_height / patch_size) * ((long long)target_width / patch_size);

        if (num_patches > max_num_patches) {
            scale -= 0.02f;
        } else {
            break;
        }
    }
}

/**
 * @brief 将 BGR 图片转换为 Vision Transformer 输入所需的 Patch Embeddings
 *
 * 逻辑：
 * 1. 按照 16x16 切分图片。
 * 2. 如果图片边缘不足 16，则视为补 0（黑色），归一化后值为 -1.0。
 * 3. 数据格式：(num_patches, 768)。每一行存放一个 Patch 的数据。
 * 4. Patch 内部排列：先放 16x16 的 B 通道，再放 G，再放 R (Planar 格式)。
 *
 * @param bgr 输入的 cv::Mat (BGR 格式, uint8)
 * @return ncnn::Mat 输出矩阵, 维度 [w=768, h=num_patches]
 */
static ncnn::Mat bgr_to_pixel_values(const cv::Mat& bgr)
{
    const int patch_size = 16;

    const float image_mean[3] = {0.5, 0.5, 0.5};
    const float image_std[3] = {0.5, 0.5, 0.5};

    int img_h = bgr.rows;
    int img_w = bgr.cols;

    // 1. 计算 grid 尺寸 (向上取整)
    int num_patches_h = (img_h + patch_size - 1) / patch_size;
    int num_patches_w = (img_w + patch_size - 1) / patch_size;
    int num_patches = num_patches_h * num_patches_w;

    // 2. 创建输出 Mat
    // 维度: w = 16*16*3 = 768, h = num_patches
    // NCNN Mat(w, h) 构造函数
    int embed_dim = patch_size * patch_size * 3;
    ncnn::Mat pixel_values(embed_dim, num_patches);

    // 3. 遍历每个 Patch
    // 使用 OpenMP 加速 (如果 NCNN 开启了 OpenMP，这里也可以手动加)
    // #pragma omp parallel for
    for (int p = 0; p < num_patches; p++)
    {
        // 计算当前 Patch 在 Grid 中的坐标
        int ph = p / num_patches_w;
        int pw = p % num_patches_w;

        // 计算当前 Patch 在原图中的像素起始坐标
        int start_y = ph * patch_size;
        int start_x = pw * patch_size;

        // 获取输出 Mat 当前行的指针
        float* out_ptr = pixel_values.row(p);

        // 指针偏移量：因为是 Planar 格式，B/G/R 分开存
        // [BBBB... GGGG... RRRR...]
        float* ptr_r = out_ptr;
        float* ptr_g = out_ptr + patch_size * patch_size;
        float* ptr_b = out_ptr + patch_size * patch_size * 2;

        // 遍历 Patch 内部 16x16 像素
        for (int y = 0; y < patch_size; y++) {
            // 预计算当前行的图像指针（如果在图像范围内）
            const uchar* img_row_ptr = NULL;
            int cur_img_y = start_y + y;
            if (cur_img_y < img_h) {
                img_row_ptr = bgr.ptr<uchar>(cur_img_y);
            }

            for (int x = 0; x < patch_size; x++) {
                int cur_img_x = start_x + x;

                // 检查边界
                if (img_row_ptr && cur_img_x < img_w) {
                    // 读取 BGR 像素
                    // cv::Mat 默认是 BGR 顺序: [b, g, r]
                    const uchar* pixel = img_row_ptr + cur_img_x * 3;
                    uchar b = pixel[0];
                    uchar g = pixel[1];
                    uchar r = pixel[2];

                    // 归一化并写入
                    *ptr_r++ = (r / 255.5f - image_mean[0]) / image_std[0];
                    *ptr_g++ = (g / 255.5f - image_mean[1]) / image_std[1];
                    *ptr_b++ = (b / 255.5f - image_mean[2]) / image_std[2];
                } else {
                    // Padding (补 0)
                    float pad_val = 0.0f;
                    *ptr_r++ = pad_val;
                    *ptr_g++ = pad_val;
                    *ptr_b++ = pad_val;
                }
            }
        }
    }

    return pixel_values;
}

static ncnn::Mat reorder_patches_for_merge(const ncnn::Mat& pixel_values, int h_patches, int w_patches, int merge_size = 2) {
    int num_patches = pixel_values.h;
    int feature_dim = pixel_values.w; // 768

    // 检查输入维度是否合法
    if (num_patches != h_patches * w_patches) {
        fprintf(stderr, "Error: h_patches * w_patches (%d * %d) != pixel_values.h (%d)\n", h_patches, w_patches, num_patches);
        return ncnn::Mat();
    }

    // 计算 Block 的网格尺寸
    // 例如 h=22, w=30, merge=2 -> grid_h=11, grid_w=15
    int grid_h = h_patches / merge_size;
    int grid_w = w_patches / merge_size;

    // 创建输出 Mat
    ncnn::Mat reordered_pixel_values(feature_dim, num_patches);

    int new_row_idx = 0;

    // 1. 遍历 Block 行
    for (int gh = 0; gh < grid_h; gh++) {
        // 2. 遍历 Block 列
        for (int gw = 0; gw < grid_w; gw++) {

            // 3. 遍历 Block 内部 (2x2)
            // 顺序通常是：Block内第一行，Block内第二行...
            for (int mh = 0; mh < merge_size; mh++) {
                for (int mw = 0; mw < merge_size; mw++) {

                    // 计算原始 Patch 的坐标
                    int original_h = gh * merge_size + mh;
                    int original_w = gw * merge_size + mw;

                    // 计算原始 Patch 在 pixel_values 中的行索引 (Row-Major)
                    int original_row_idx = original_h * w_patches + original_w;

                    // 获取源指针和目标指针
                    const float* src_ptr = pixel_values.row(original_row_idx);
                    float* dst_ptr = reordered_pixel_values.row(new_row_idx);

                    // 内存复制 (整行复制，速度快)
                    memcpy(dst_ptr, src_ptr, feature_dim * sizeof(float));

                    new_row_idx++;
                }
            }
        }
    }

    // 如果原始图片尺寸不能被 merge_size 整除（例如 h=23），上述循环会忽略边缘。
    // 如果你的模型逻辑包含 Padding，请确保 h_patches/w_patches 已经是 Padding 后的尺寸。
    // 如果需要处理边缘剩余 Patch，需要额外的逻辑，但通常 Vision Transformer 会先 Pad 到整除。

    return reordered_pixel_values;
}

int qwen3_5_0p8b::get_visiual_features(const cv::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h) const
{
    if (bgr.empty())
    {
        image_embeds.release();
        num_patches_w = 0;
        num_patches_h = 0;
        return 0;
    }

    const int patch_size = 16;
    const int max_num_patches = 49152;

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;
    fprintf(stderr, "image %d %d\n", img_w, img_h);

    // get_image_size_for_patches
    int target_w;
    int target_h;
    get_image_size_for_patches(img_h, img_w, patch_size, max_num_patches, target_h, target_w);
    fprintf(stderr, "target %d %d\n", target_w, target_h);

    // resize
    cv::Mat bgr_resized;
    {
        // FIXME this is not pil image resize
        cv::resize(bgr, bgr_resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_AREA);
    }

    num_patches_w = (target_w + patch_size - 1) / patch_size;
    num_patches_h = (target_h + patch_size - 1) / patch_size;
    fprintf(stderr, "patches %d %d\n", num_patches_w, num_patches_h);

    const int seq_len = num_patches_w * num_patches_h;
    fprintf(stderr, "seq_len %d\n", seq_len);

    ncnn::Mat pixel_values = bgr_to_pixel_values(bgr_resized);
    // print_mat(pixel_values);

    pixel_values = reorder_patches_for_merge(pixel_values, num_patches_h, num_patches_w, 2);

    pixel_values = pixel_values.reshape(16 * 16, 1, 3, seq_len);
    // dup depth
    {
        // TODO we can modify embed_patch weights to eliminate this depth duplication
        ncnn::Mat tmp(16 * 16, 2, 3, seq_len);
        for (int i = 0; i < seq_len; i++)
        {
            memcpy(tmp.channel(i).depth(0).row(0), pixel_values.channel(i).depth(0).row(0), 16 * 16 * sizeof(float));
            memcpy(tmp.channel(i).depth(0).row(1), pixel_values.channel(i).depth(0).row(0), 16 * 16 * sizeof(float));
            memcpy(tmp.channel(i).depth(1).row(0), pixel_values.channel(i).depth(1).row(0), 16 * 16 * sizeof(float));
            memcpy(tmp.channel(i).depth(1).row(1), pixel_values.channel(i).depth(1).row(0), 16 * 16 * sizeof(float));
            memcpy(tmp.channel(i).depth(2).row(0), pixel_values.channel(i).depth(2).row(0), 16 * 16 * sizeof(float));
            memcpy(tmp.channel(i).depth(2).row(1), pixel_values.channel(i).depth(2).row(0), 16 * 16 * sizeof(float));
        }

        pixel_values = tmp.reshape(16 * 16 * 2 * 3, seq_len);
    }
    // print_mat(pixel_values);

    // 3. First Network: vision_embed
    ncnn::Mat patch_embeds(768, seq_len);
    for (int i = 0; i < seq_len; i++)
    {
        ncnn::Mat patch = pixel_values.row_range(i, 1).reshape(16, 16, 2, 3);
        ncnn::Mat patch_embed;

        ncnn::Extractor ex = impl_->vision_embed_patch.create_extractor();
        ex.input("in0", patch);
        ex.extract("out0", patch_embed);

        memcpy(patch_embeds.row(i), patch_embed.reshape(768), 768 * sizeof(float));
    }
    // print_mat(patch_embeds);

    ncnn::Mat pos_embeds;
    {
        ncnn::Mat grid(num_patches_w, num_patches_h);

        ncnn::Extractor ex = impl_->vision_embed_pos.create_extractor();
        ex.input("in0", grid);
        ex.extract("out0", pos_embeds);
    }
    // print_mat(pos_embeds);

    pos_embeds = reorder_patches_for_merge(pos_embeds, num_patches_h, num_patches_w, 2);

    // print_mat(pos_embeds);

    // 4. Emb Cos/Sin
    ncnn::Mat emb_cos, emb_sin;
    generate_rope_embeds(num_patches_w, num_patches_h, emb_cos, emb_sin, 32);

    // print_mat(emb_cos);
    // print_mat(emb_sin);

    // 7. Second Network: vision_encoder
    {
        ncnn::Extractor ex = impl_->vision_encoder.create_extractor();
        ex.input("in0", patch_embeds);
        ex.input("in1", pos_embeds);
        ex.input("in2", emb_cos);
        ex.input("in3", emb_sin);

        ex.extract("out0", image_embeds);
    }

    // print_mat(image_embeds);

    return 0;
}
