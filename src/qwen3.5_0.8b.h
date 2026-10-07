// Copyright 2026 Tencent
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cassert>
#include <functional>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <opencv2/core/core.hpp>

#include "mat.h"

struct GenerateConfig {
    int max_new_tokens = 8192;
    float temperature = 1.f;
    float top_p = 1.f;
    int top_k = 50;
    float repetition_penalty = 1.05f;
    int beam_size = 1;
    int do_sample = 0;
};

struct qwen3_5_0p8b_ctx;

class qwen3_5_0p8b {
public:
    qwen3_5_0p8b(
              std::string vision_embed_patch_param,
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
              bool use_vulkan);

    ~qwen3_5_0p8b();

    std::shared_ptr<qwen3_5_0p8b_ctx> prefill(const std::string& input_text) const;

    std::shared_ptr<qwen3_5_0p8b_ctx> prefill(const std::string& input_text, const cv::Mat& bgr, const std::shared_ptr<qwen3_5_0p8b_ctx> ctx) const;

    std::shared_ptr<qwen3_5_0p8b_ctx> prefill(const std::string& input_text, const std::shared_ptr<qwen3_5_0p8b_ctx> ctx) const;

    std::shared_ptr<qwen3_5_0p8b_ctx> generate(const std::shared_ptr<qwen3_5_0p8b_ctx>& ctx, const GenerateConfig& cfg, std::function<void(const std::string&)> callback) const;

    bool decode(std::shared_ptr<qwen3_5_0p8b_ctx> ctx, std::function<void(const std::string&)> callback) const;

    // for vision encoder
    int get_visiual_features(const cv::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
