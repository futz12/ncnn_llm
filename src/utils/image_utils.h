#pragma once

#include <string>
#include <mat.h>

ncnn::Mat load_image_to_ncnn_mat(const std::string& image_path);

ncnn::Mat ncnn_mat_resize(const ncnn::Mat& src, int target_w, int target_h);

ncnn::Mat ncnn_mat_resize_bicubic(const ncnn::Mat& src, int target_w, int target_h);

bool ncnn_mat_empty(const ncnn::Mat& mat);

#include <cstdint>
#include <vector>
namespace ncnn_llm { namespace minicpm {
struct Thumbnail {
    int height, width, patch_rows, patch_cols, token_rows, token_cols;
    int token_count() const { return token_rows * token_cols; }
};
// Matches HF MiniCPMV4_6ImageProcessor (also used for 4.7). Resize pixels
// separately with the checkpoint's bicubic backend; this function plans dimensions.
// Rejects cases where enabled slicing would produce detail crops.
Thumbnail plan_thumbnail(int height, int width, int patch_size, int scale_resolution,
                         const std::string& mode, bool slice_mode = true, int max_slices = 9);
int merge_factor(const std::string& mode);
// Input: already resized, interleaved RGB uint8. Output: contiguous [3,p,H*W/p]
// NaViT layout with rescale=1/255, mean=std=0.5. No implicit resizing.
std::vector<float> pack_rgb(const std::vector<uint8_t>& rgb, int height, int width, int patch_size);
std::vector<int> nearest_position_ids(int patch_rows, int patch_cols, int learned_side);
// Full-image RGB uint8 resize matching Pillow 12.3.0 BICUBIC, no reducing_gap.
// Antialiased on downsampling; tightly packed three-channel input/output.
// input_bgr converts BGR input to RGB output. No crop boxes, alpha or float input.
std::vector<uint8_t> resize_bicubic_rgb(const std::vector<uint8_t>& pixels,
    int height, int width, int target_height, int target_width, bool input_bgr=false);
} }
