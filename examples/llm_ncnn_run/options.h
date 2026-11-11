#pragma once

#include <string>

struct Options {
    std::string model_path = "./assets/qwen3_0.6b";
    std::string image_path;
    bool use_vulkan = false;
    bool use_bf16 = true;
    bool enable_builtin_tools = true;
    bool enable_thinking = false;
    int num_threads = 0;  // 0 = use ncnn::get_cpu_count()
    int vulkan_device = 0;  // Vulkan device index
    int max_new_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repetition_penalty = 1.1f;
    bool do_sample = false;
    bool enable_perf = false;
    int perf_level = 0;
    bool force_naive = false;  // use naive GDR/ShortConv instead of x86 SIMD
};

Options parse_options(int argc, char** argv);
