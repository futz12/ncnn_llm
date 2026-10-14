#pragma once

#include <mat.h>
#include <vector>

struct RopeScalingParams {
    float alpha;
    float beta_fast;
    float beta_slow;
    float factor;
    float mscale;
    float mscale_all_dim;
};

void generate_ntk_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
);

void generate_yarn_rope_embed_cache(
    int seqlen, 
    int embed_dim, 
    int position_id, 
    ncnn::Mat& cos_cache, 
    ncnn::Mat& sin_cache, 
    float rope_theta, 
    const RopeScalingParams& scaling_params
);

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta = 100000);

void generate_rope_embed_cache_full(int seqlen, int head_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta = 100000);

void generate_rope_embed_cache_LongRoPE(int seqlen,
                                      int embed_dim,
                                      int position_id,
                                      ncnn::Mat& cos_cache,
                                      ncnn::Mat& sin_cache,
                                      float rope_theta,
                                      const float* SHORT_FACTOR,
                                      const float* LONG_FACTOR,
                                      int ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768);

// Legacy single-placeholder interface retained for existing callers.
void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, int image_pad_id, const ncnn::Mat& image_embeds);
// Pre-expanded image spans keep prompt tokenization intact (e.g. MiniCPM canvas).
void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, int image_pad_id, const ncnn::Mat& image_embeds, bool expanded_placeholder_span);

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
                                          float rope_theta = 100000);

void generate_rope_embed_cache_vision_mrope_interleaved(int seqlen,
                                                        int embed_dim,
                                                        int position_id,
                                                        int image_pad_index,
                                                        int image_embeds_size,
                                                        int num_patches_w,
                                                        int num_patches_h,
                                                        ncnn::Mat& cos_cache,
                                                        ncnn::Mat& sin_cache,
                                                        float rope_theta = 100000);

// HunyuanOCR 4-axis xdrope. pos4 = { linear, w, h, t }, each of length seq_len.
// Produces cos/sin caches of shape (rope_head_dim/2, seq_len) with an NTK-alpha base:
//   base = rope_theta * alpha^(rope_head_dim/(rope_head_dim-2));
//   inv_freq[j] = 1 / base^(2j/rope_head_dim);
// dim j uses axis = xdrope_section prefix (e.g. [16,16,16,16] -> j/16).
void generate_hunyuan_xdrope_cos_sin(const std::vector<int>* pos4,
                                     int seq_len,
                                     int rope_head_dim,
                                     const std::vector<int>& xdrope_section,
                                     float rope_theta,
                                     float alpha,
                                     ncnn::Mat& cos_cache,
                                     ncnn::Mat& sin_cache);
// 3D RoPE cache (w = head_dim / 2, h = seqlen, c = 1) for models like ModernBERT / Laya
void generate_rope_embed_cache_3d(int seqlen,
                                  int head_dim,
                                  float rope_theta,
                                  ncnn::Mat& cos_cache,
                                  ncnn::Mat& sin_cache);


#include <array>
#include <cstdint>
#include <string>
namespace ncnn_llm { namespace minicpm {
struct MarkerIds {
    int image, image_start, image_end, slice_start, slice_end, newline;
};
struct CanvasPositions {
    // One {T,H,W} tuple per sequence token; not the separate causal text channel.
    std::vector<std::array<int64_t, 3> > positions;
    int64_t rope_delta;
    int64_t next_position;
};
// Exactly one image, explicit start/end markers, no padding or video/slices.
// base_position is a scalar offset for an already-computed chat prefix.
CanvasPositions canvas_positions(const std::vector<int>& ids, const MarkerIds& markers,
                                  int patch_rows, int patch_cols, const std::string& mode,
                                  int64_t base_position = 0);
// Row-major [sequence,rotary_dim/2] cache for ncnn RotaryEmbed. Sections are
// checkpoint mrope_section values; do not hardcode Qwen defaults in the caller.
void interleaved_rope(const CanvasPositions& positions, int rotary_dim, double theta,
                      const std::array<int,3>& sections,
                      std::vector<float>& cos_cache, std::vector<float>& sin_cache);
} }
