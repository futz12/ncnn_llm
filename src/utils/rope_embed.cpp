#include "rope_embed.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdio>

static float yarn_ramp(float low, float high, float val) {
    if (val < low) return 0.0f;
    if (val > high) return 1.0f;
    return (val - low) / (high - low);
}

void generate_yarn_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
) {
    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    std::vector<float> inv_freq(embed_dim / 2);
    
    float scale = scaling_params.alpha; 
    float beta_fast = scaling_params.beta_fast;
    float beta_slow = scaling_params.beta_slow;

    for (int i = 0; i < embed_dim / 2; i++) {
        float freq = 1.0f / powf(rope_theta, (float)(i * 2) / (float)embed_dim);
        
        float r = (float)(i * 2) / (float)embed_dim;
        
        float ramp = yarn_ramp(beta_slow, beta_fast, (float)embed_dim / (i * 2 + 1)); 
        float base_correction = powf(scale, (float)embed_dim / ((float)embed_dim - 2.0f));
        float ntk_theta = rope_theta * base_correction;
        inv_freq[i] = 1.0f / powf(ntk_theta, (float)(i * 2) / (float)embed_dim);
    }

    for (int i = 0; i < seqlen; i++) {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++) {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            
            float mscale = scaling_params.mscale;
            if (fabs(mscale - 1.0f) > 1e-6) {
                *cos_ptr++ = cosf(t) * mscale;
                *sin_ptr++ = sinf(t) * mscale;
            } else {
                *cos_ptr++ = cosf(t);
                *sin_ptr++ = sinf(t);
            }
        }
    }
}

void generate_ntk_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
) {
    float dim = (float)embed_dim;
    float alpha = scaling_params.alpha;
    
    float base_correction = powf(alpha, dim / (dim - 2.0f));
    float ntk_theta = rope_theta * base_correction;

    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(ntk_theta, (float)(i * 2) / dim);
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
            
            *cos_ptr++ = cosf(t);
            *sin_ptr++ = sinf(t);
        }
    }
}

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta)
{
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

void generate_rope_embed_cache_full(int seqlen, int head_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta)
{
    int half_dim = head_dim / 2;
    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / head_dim);
    }

    cos_cache.create(head_dim, seqlen);
    sin_cache.create(head_dim, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < half_dim; j++)
        {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            cos_ptr[j] = cos_val;
            cos_ptr[j + half_dim] = cos_val;
            sin_ptr[j] = sin_val;
            sin_ptr[j + half_dim] = sin_val;
        }
    }
}

void generate_rope_embed_cache_vision_mrope_interleaved(int seqlen,
                                                        int embed_dim,
                                                        int position_id,
                                                        int image_pad_index,
                                                        int image_embeds_size,
                                                        int num_patches_w,
                                                        int num_patches_h,
                                                        ncnn::Mat& cos_cache,
                                                        ncnn::Mat& sin_cache,
                                                        float rope_theta)
{
    const int merge_size = 2;
    const int mrope[3] = {11, 11, 10};

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
            int which_pos = 0;
            if (j < mrope[1] * 3 && (j % 3 == 1))
            {
                which_pos = 1;
            }
            else if (j < mrope[2] * 3 && (j % 3 == 2))
            {
                which_pos = 2;
            }

            int pos = position_id;
            if (i < image_pad_index)
            {
                pos += i;
            }
            else if (i >= image_pad_index + image_embeds_size)
            {
                pos += i - image_embeds_size +
                       (std::max(num_patches_w, num_patches_h) / merge_size);
            }
            else if (which_pos == 0)
            {
                pos += image_pad_index;
            }
            else if (which_pos == 1)
            {
                int hid = (i - image_pad_index) / (num_patches_w / merge_size);
                pos += image_pad_index + hid;
            }
            else
            {
                int wid = (i - image_pad_index) % (num_patches_w / merge_size);
                pos += image_pad_index + wid;
            }

            const float t = pos * inv_freq[j];
            *cos_ptr++ = cosf(t);
            *sin_ptr++ = sinf(t);
        }
    }
}

static inline float compute_scaling_factor(int max_position_embeddings, int ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768) {
    float scale = static_cast<float>(max_position_embeddings) / static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS);
    return std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS)));
}

void generate_rope_embed_cache_LongRoPE(int seqlen,
                                      int embed_dim,
                                      int position_id,
                                      ncnn::Mat& cos_cache,
                                      ncnn::Mat& sin_cache,
                                      float rope_theta,
                                      const float* SHORT_FACTOR,
                                      const float* LONG_FACTOR,
                                      int ORIGINAL_MAX_POSITION_EMBEDDINGS)
{
    if (embed_dim % 2 != 0 || seqlen <= 0) {
        cos_cache.release();
        sin_cache.release();
        return;
    }

    const int half_dim = embed_dim / 2;

    cos_cache.create(half_dim, seqlen);
    sin_cache.create(half_dim, seqlen);

    if (cos_cache.empty() || sin_cache.empty()) {
        return;
    }

    float* cos_ptr = cos_cache.channel(0);
    float* sin_ptr = sin_cache.channel(0);

    std::vector<float> inv_freq(half_dim);
    for (int j = 0; j < half_dim; ++j) {
        float exponent = (2.0f * j) / static_cast<float>(embed_dim);
        inv_freq[j] = 1.0f / std::pow(rope_theta, exponent);
    }

    const float* ext_factor = (seqlen > ORIGINAL_MAX_POSITION_EMBEDDINGS) ? LONG_FACTOR : SHORT_FACTOR;
    const float scaling_factor = compute_scaling_factor(ORIGINAL_MAX_POSITION_EMBEDDINGS);

    for (int i = 0; i < seqlen; ++i) {
        int t = position_id + i;
        float* row_cos = cos_ptr + i * half_dim;
        float* row_sin = sin_ptr + i * half_dim;

        for (int j = 0; j < half_dim; ++j) {
            float freq = (static_cast<float>(t) * inv_freq[j]) / ext_factor[j];
            row_cos[j] = std::cos(freq) * scaling_factor;
            row_sin[j] = std::sin(freq) * scaling_factor;
        }
    }
}

void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, int image_pad_id, const ncnn::Mat& image_embeds)
{
    inject_image_embeds(token_ids, token_embed, image_pad_index, image_pad_id, image_embeds, false);
}

void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, int image_pad_id, const ncnn::Mat& image_embeds, bool expanded_placeholder_span)
{
    image_pad_index = -1;
    if (image_embeds.empty() || token_ids.empty() || token_embed.empty() || image_pad_id < 0 ||
        token_embed.h != (int)token_ids.size() || token_embed.w != image_embeds.w)
    {
        return;
    }

    int placeholder_index = -1;
    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        if (token_ids[i] == image_pad_id)
        {
            placeholder_index = i;
            break;
        }
    }
    if (placeholder_index < 0)
    {
        return;
    }

    if (expanded_placeholder_span) {
        if (image_embeds.h > (int)token_ids.size() - placeholder_index) return;
        for (int i = 0; i < image_embeds.h; ++i)
            if (token_ids[placeholder_index + i] != image_pad_id) return;
        image_pad_index = placeholder_index;
        for (int i = 0; i < image_embeds.h; ++i)
            memcpy(token_embed.row(placeholder_index + i), image_embeds.row(i), token_embed.w * sizeof(float));
        return;
    }

    std::vector<int> token_ids_injected(token_ids.size() - 1 + image_embeds.h);
    ncnn::Mat token_embed_injected(token_embed.w, token_embed.h - 1 + image_embeds.h);
    if (token_embed_injected.empty()) return;

    image_pad_index = placeholder_index;

    memcpy(token_ids_injected.data(), token_ids.data(), placeholder_index * sizeof(int));
    std::fill_n(token_ids_injected.data() + placeholder_index, image_embeds.h, image_pad_id);
    memcpy(token_ids_injected.data() + placeholder_index + image_embeds.h,
           token_ids.data() + placeholder_index + 1,
           (token_ids.size() - 1 - placeholder_index) * sizeof(int));

    memcpy(token_embed_injected.row(0), token_embed.row(0), placeholder_index * token_embed.w * sizeof(float));
    memcpy(token_embed_injected.row(placeholder_index), image_embeds.row(0), image_embeds.h * token_embed.w * sizeof(float));
    if (placeholder_index + 1 < (int)token_ids.size()) {
        memcpy(token_embed_injected.row(placeholder_index + image_embeds.h),
               token_embed.row(placeholder_index + 1),
               (token_ids.size() - 1 - placeholder_index) * token_embed.w * sizeof(float));
    }

    token_ids = token_ids_injected;
    token_embed = token_embed_injected;
}

void generate_rope_embed_cache_vision_mrope(int seqlen, 
                                          int embed_dim, 
                                          int position_id, 
                                          int image_pad_index, 
                                          int image_embeds_size, 
                                          int num_patches_w, 
                                          int spatial_merge_size,
                                          const std::vector<int>& mrope_section,
                                          ncnn::Mat& cos_cache, 
                                          ncnn::Mat& sin_cache, 
                                          float rope_theta)
{
    if (mrope_section.size() < 3) return;

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
            int pos = position_id;
            if (i < image_pad_index)
            {
                pos += i;
            }
            else if (i >= image_pad_index + image_embeds_size)
            {
                pos += i - image_embeds_size + (num_patches_w / spatial_merge_size);
            }
            else
            {
                // inside image tokens range
                if (j < mrope_section[0])
                {
                    // temporal
                    pos += image_pad_index;
                }
                else if (j < mrope_section[0] + mrope_section[1])
                {
                    // height
                    int hid = (i - image_pad_index) / (num_patches_w / spatial_merge_size);
                    pos += image_pad_index + hid;
                }
                else
                {
                    // width
                    int wid = (i - image_pad_index) % (num_patches_w / spatial_merge_size);
                    pos += image_pad_index + wid;
                }
            }

            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            *cos_ptr++ = cos_val;
            *sin_ptr++ = sin_val;
        }
    }
}
void generate_hunyuan_xdrope_cos_sin(const std::vector<int>* pos4,
                                     int seq_len,
                                     int rope_head_dim,
                                     const std::vector<int>& xdrope_section,
                                     float rope_theta,
                                     float alpha,
                                     ncnn::Mat& cos_cache,
                                     ncnn::Mat& sin_cache) {
    const int half = rope_head_dim / 2;
    cos_cache.create(half, seq_len);
    sin_cache.create(half, seq_len);

    // NTK-alpha base: base = rope_theta * alpha^(head_dim/(head_dim-2))
    const double base_corr = std::pow((double)alpha,
                                      (double)rope_head_dim / ((double)rope_head_dim - 2.0));
    const double ntk_theta = (double)rope_theta * base_corr;

    std::vector<float> inv_freq(half);
    for (int j = 0; j < half; j++) {
        inv_freq[j] = (float)(1.0 / std::pow(ntk_theta, (double)(2 * j) / (double)rope_head_dim));
    }

    // axis index per dim, from the (possibly unequal) section widths
    std::vector<int> axis_of_dim(half, (int)xdrope_section.size() - 1);
    {
        int d = 0;
        for (int a = 0; a < (int)xdrope_section.size() && d < half; a++) {
            for (int s = 0; s < xdrope_section[a] && d < half; s++) {
                axis_of_dim[d++] = a;
            }
        }
    }

    for (int i = 0; i < seq_len; i++) {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);
        for (int j = 0; j < half; j++) {
            const int a = axis_of_dim[j];
            const float pos = (float)pos4[a][i];
            const float t = pos * inv_freq[j];
            cos_ptr[j] = cosf(t);
            sin_ptr[j] = sinf(t);
        }
    }
}

void generate_rope_embed_cache_3d(int seqlen,
                                  int head_dim,
                                  float rope_theta,
                                  ncnn::Mat& cos_cache,
                                  ncnn::Mat& sin_cache)
{
    const int half_dim = head_dim / 2;
    cos_cache.create(half_dim, seqlen, 1, sizeof(float));
    sin_cache.create(half_dim, seqlen, 1, sizeof(float));

    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        inv_freq[i] = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }

    for (int i = 0; i < seqlen; ++i)
    {
        float* r_cos = cos_cache.row(i);
        float* r_sin = sin_cache.row(i);
        for (int j = 0; j < half_dim; ++j)
        {
            const float angle = static_cast<float>(i) * inv_freq[j];
            r_cos[j] = std::cos(angle);
            r_sin[j] = std::sin(angle);
        }
    }
}


#include "image_utils.h"
#include <stdexcept>
namespace ncnn_llm { namespace minicpm {
namespace {
void require(bool ok, const char* reason) { if (!ok) throw std::invalid_argument(reason); }
}
CanvasPositions canvas_positions(const std::vector<int>& ids, const MarkerIds& m,
                                  int h, int w, const std::string& mode, int64_t base) {
    const int merge=merge_factor(mode);
    require(h>0 && w>0 && h%merge==0 && w%merge==0 && base>=0, "invalid canvas dimensions or offset");
    const int64_t count=static_cast<int64_t>(h/merge)*(w/merge);
    size_t start=ids.size(); int starts=0, ends=0;
    for(size_t i=0;i<ids.size();++i) {
        require(ids[i]!=m.slice_start && ids[i]!=m.slice_end, "MiniCPM detail slices are unsupported");
        if(ids[i]==m.image_start) { start=i; ++starts; }
        if(ids[i]==m.image_end) ++ends;
    }
    require(starts==1 && ends==1, "exactly one explicitly marked MiniCPM image is required");
    require(count<=static_cast<int64_t>(ids.size()) && start+1+count<ids.size(), "visual token count mismatch");
    const size_t end=start+1+static_cast<size_t>(count);
    require(ids[end]==m.image_end, "missing image end marker or visual token count mismatch");
    for(size_t i=0;i<ids.size();++i)
        require((ids[i]==m.image)==(i>start && i<end), "visual placeholder span mismatch");
    CanvasPositions out;
    out.positions.resize(ids.size());
    for(size_t i=0;i<start;++i) out.positions[i]={{base+static_cast<int64_t>(i),base+static_cast<int64_t>(i),base+static_cast<int64_t>(i)}};
    const int64_t origin=base+start;
    out.positions[start]={{origin,std::max<int64_t>(origin-1,0),std::max<int64_t>(origin-1,0)}};
    for(int y=0;y<h/merge;++y) for(int x=0;x<w/merge;++x)
        out.positions[start+1+static_cast<size_t>(y)*(w/merge)+x]={{origin,origin+y,origin+x}};
    out.positions[end]={{origin,origin+h/merge,origin+w/merge}};
    // HF groups all trailing marker/newline tokens into the frame; newlines
    // immediately after </image> inherit the canvas origin.
    size_t trailing=end+1;
    while(trailing<ids.size() && ids[trailing]==m.newline) out.positions[trailing++]={{origin,origin,origin}};
    int64_t next=origin+std::max(h/merge,w/merge)+1;
    for(size_t i=trailing;i<ids.size();++i,++next) out.positions[i]={{next,next,next}};
    out.next_position=next;
    out.rope_delta=next-static_cast<int64_t>(ids.size());
    return out;
}
void interleaved_rope(const CanvasPositions& p, int dim, double theta,
                      const std::array<int,3>& sections, std::vector<float>& cc, std::vector<float>& ss) {
    require(dim>0 && dim%2==0 && std::isfinite(theta) && theta>0, "invalid rotary dimensions/theta");
    require(sections[0]>=0 && sections[1]>=0 && sections[2]>=0 &&
            static_cast<int64_t>(sections[0])+sections[1]+sections[2]==dim/2,
            "mrope sections must sum to rotary_dim/2");
    cc.resize(p.positions.size()*(dim/2)); ss.resize(cc.size());
    for(size_t i=0;i<p.positions.size();++i) for(int j=0;j<dim/2;++j) {
        int axis=(j%3==1 && j<3*sections[1])?1:((j%3==2 && j<3*sections[2])?2:0);
        const double angle=p.positions[i][axis]/std::pow(theta,static_cast<double>(2*j)/dim);
        cc[i*(dim/2)+j]=static_cast<float>(std::cos(angle));
        ss[i*(dim/2)+j]=static_cast<float>(std::sin(angle));
    }
}
} } // namespace ncnn_llm::minicpm
