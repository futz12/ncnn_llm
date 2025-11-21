#include "minicpm4_0.5b.h"

#include <memory>
#include <ncnn/mat.h>
#include <ncnn/net.h>
#include <utility>
#include <vector>

#include "utils/tokenizer/bpe_tokenizer.h"

const static int attn_cnt = 24;

struct minicpm4_0_5b_ctx {
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;

    int cur_token = 0;
};

class minicpm4_0_5b::Impl {
public:
    ncnn::Net embed_net;
    ncnn::Net proj_out_net;
    ncnn::Net decoder_net;

    BpeTokenizer bpe;

    Impl(std::string embed_param,
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
              SpecialTokensConfig{
                  .bos_token = "<s>",
                  .eos_token = "</s>",
                  .unk_token = "<unk>",
                  .sep_token = "<SEP>",
                  .cls_token = "<CLS>",
                  .mask_token = "<mask>",
              })) {
        if (use_vulkan) {
            embed_net.opt.use_vulkan_compute = true;
            proj_out_net.opt.use_vulkan_compute = true;
            decoder_net.opt.use_vulkan_compute = true;
        }
        embed_net.load_param(embed_param.c_str());
        embed_net.load_model(embed_bin.c_str());
        proj_out_net.load_param(proj_out_param.c_str());
        proj_out_net.load_model(embed_bin.c_str());
        decoder_net.load_param(decoder_param.c_str());
        decoder_net.load_model(decoder_bin.c_str());
    }
};

// ref transformers/modeling_rope_utils.py _compute_default_rope_parameters()
// ref transformers/models/llama/modeling_llama.py class LlamaRotaryEmbedding
static void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache)
{
    // rope related hyper-parameters from config.json
    const float rope_theta = 100000;
    const float attention_factor = 1.f;

    // prepare inv_freq
    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / embed_dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            *cos_ptr++ = cos_val;
            *sin_ptr++ = sin_val;
        }
    }
}

minicpm4_0_5b::minicpm4_0_5b(std::string embed_param,
                                 std::string embed_bin,
                                 std::string proj_out_param,
                                 std::string decoder_param,
                                 std::string decoder_bin,
                                 std::string vocab_file,
                                 std::string merges_file,
                                 bool use_vulkan)
    : impl_(std::make_unique<Impl>(std::move(embed_param),
                                  std::move(embed_bin),
                                  std::move(proj_out_param),
                                  std::move(decoder_param),
                                  std::move(decoder_bin),
                                  std::move(vocab_file),
                                  std::move(merges_file),
                                  use_vulkan)) {
    
}

minicpm4_0_5b::~minicpm4_0_5b() = default;

std::shared_ptr<minicpm4_0_5b_ctx> minicpm4_0_5b::prefill(const std::string& input_text) {
    auto token_ids = impl_->bpe.encode(input_text, true, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_rope_embed_cache(token_ids.size(), 32, 0, cos_cache, sin_cache);

    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data());
    ncnn::Mat token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", input_ids_mat);
        ex.extract("out0", token_embed);
    }

    /*
    ncnn::Mat token_embed(1024, src-seqlen);
    ncnn::Mat mask(cur-seqlen, src-seqlen);
    ncnn::Mat cos_cache(32, cur-seqlen);
    ncnn::Mat sin_cache(32, cur-seqlen);
    */

    ncnn::Mat mask(token_ids.size(), token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        float* row = mask.row(i);
        for (int j = i + 1; j < (int)token_ids.size(); j++) {
            row[j] = -10000.0f;
        }
    }

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;
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
    }

    // full process last token
    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id);
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = impl_->embed_net.create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    ncnn::Mat last_cos_cache;
    ncnn::Mat last_sin_cache;
    generate_rope_embed_cache(1, 32, token_ids.size(), last_cos_cache, last_sin_cache);
    ncnn::Mat last_mask(1, token_ids.size() + 1);
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

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
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

    auto ctx = std::make_shared<minicpm4_0_5b_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;

    return ctx;
}

std::shared_ptr<minicpm4_0_5b_ctx> minicpm4_0_5b::prefill(const std::string& input_text,
                                                 const std::shared_ptr<minicpm4_0_5b_ctx> ctx) {
    return nullptr;
}

bool minicpm4_0_5b::decode(std::shared_ptr<minicpm4_0_5b_ctx> ctx,
                            std::function<void(const std::string&)> callback) {

    while (ctx->cur_token != impl_->bpe.special_ids().eos_id) {
        callback(impl_->bpe.id_to_token()[ctx->cur_token]);

        ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token);
        ncnn::Mat cur_token_embed;
        {
            ncnn::Extractor ex = impl_->embed_net.create_extractor();
            ex.input("in0", cur_token_mat);
            ex.extract("out0", cur_token_embed);
        }
        ncnn::Mat cos_cache;
        ncnn::Mat sin_cache;
        generate_rope_embed_cache(1, 32, ctx->kv_cache[0].first.h, cos_cache, sin_cache);
        ncnn::Mat mask(1, ctx->kv_cache[0].first.h + 1);
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

            for (int i = 0; i < attn_cnt; i++) {
                char name_k_out[32], name_v_out[32];
                std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
                std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
                ncnn::Mat k_cache, v_cache;
                ex.extract(name_k_out, k_cache);
                ex.extract(name_v_out, v_cache);
                ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
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
        ctx->cur_token = next_token_id;
    }

    return true;
}

std::string minicpm4_0_5b::decode(std::shared_ptr<minicpm4_0_5b_ctx> ctx) {
    return {};
}

