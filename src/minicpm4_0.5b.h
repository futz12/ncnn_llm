#pragma once

#include <functional>
#include <memory>
#include <vector>
#include <string>

struct minicpm4_0_5b_ctx;

class minicpm4_0_5b {
public:
    minicpm4_0_5b(std::string embed_param,
              std::string embed_bin,
              std::string proj_out_param,
              std::string decoder_param,
              std::string decoder_bin,
              std::string vocab_file,
              std::string merges_file,
              bool use_vulkan);

    ~minicpm4_0_5b();

    std::shared_ptr<minicpm4_0_5b_ctx> prefill(const std::string& input_text);

    std::shared_ptr<minicpm4_0_5b_ctx> prefill(const std::string& input_text,
                                         const std::shared_ptr<minicpm4_0_5b_ctx> ctx);

    bool decode(std::shared_ptr<minicpm4_0_5b_ctx> ctx,
                std::function<void(const std::string&)> callback);

    std::string decode(std::shared_ptr<minicpm4_0_5b_ctx> ctx);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};