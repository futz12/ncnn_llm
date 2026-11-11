#include "options.h"

#include "util.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << (argv0 ? argv0 : "llm_ncnn_run") << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --model <path>             Model path (default: ./assets/qwen3_0.6b)\n"
        << "  --image <path>             Image path for VL models (optional)\n"
        << "  --use-vulkan               Enable Vulkan backend\n"
        << "  --no-bf16                  Disable BF16 storage (use FP32 storage)\n"
        << "  --vulkan-device <index>    Vulkan device index (default: 0)\n"
        << "  --threads <num>            Number of CPU threads (default: auto)\n"
        << "  --max-new-tokens <num>     Maximum generated tokens (default: 512)\n"
        << "  --temperature <val>        Sampling temperature (default: 0.7)\n"
        << "  --top-p <val>              Top-p sampling cutoff (default: 0.9)\n"
        << "  --top-k <num>              Top-k sampling cutoff (default: 40)\n"
        << "  --repetition-penalty <val> Repetition penalty (default: 1.1)\n"
        << "  --do-sample                Enable stochastic sampling\n"
        << "  --no-builtin-tools         Disable built-in tools (random/add)\n"
        << "  --enable-thinking          Enable model reasoning output (<think>...</think>)\n"
        << "  --perf                     Enable performance profiling report\n"
        << "  --perf-level <1|2>         Profiling detail level (1: stages, 2: stages + layers)\n"
        << "  --help                     Show this help\n"
        << "\n"
        << "Examples:\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run") << "\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run") << " --model ./assets/qwen3_0.6b\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run") << " --model ./assets/qwen3_0.6b --threads 4\n"
        << "  " << (argv0 ? argv0 : "llm_ncnn_run") << " --model ./assets/qwen2.5_vl_3b --image ./assets/test.jpg\n";
}

} // namespace

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--") {
            break;
        }
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--no-bf16") {
            opt.use_bf16 = false;
        } else if (a == "--force-naive") {
            opt.force_naive = true;
        } else if (a == "--bf16") {
            opt.use_bf16 = true;
        } else if (a == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --model\n";
                std::exit(2);
            }
            opt.model_path = argv[++i];
        } else if (a == "--image") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --image\n";
                std::exit(2);
            }
            opt.image_path = argv[++i];
        } else if (a == "--use-vulkan" || a == "--vulkan") {
            opt.use_vulkan = true;
        } else if (a == "--vulkan-device") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --vulkan-device\n";
                std::exit(2);
            }
            opt.vulkan_device = std::atoi(argv[++i]);
        } else if (a == "--threads") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --threads\n";
                std::exit(2);
            }
            opt.num_threads = std::atoi(argv[++i]);
        } else if (a == "--max-new-tokens") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --max-new-tokens\n";
                std::exit(2);
            }
            opt.max_new_tokens = std::atoi(argv[++i]);
            if (opt.max_new_tokens <= 0) opt.max_new_tokens = 512;
        } else if (a == "--no-builtin-tools") {
            opt.enable_builtin_tools = false;
        } else if (a == "--enable-thinking" || a == "--thinking") {
            opt.enable_thinking = true;
        } else if (a == "--temperature" || a == "--temp") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(2); }
            opt.temperature = std::atof(argv[++i]);
        } else if (a == "--top-p") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --top-p\n"; std::exit(2); }
            opt.top_p = std::atof(argv[++i]);
        } else if (a == "--top-k") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --top-k\n"; std::exit(2); }
            opt.top_k = std::atoi(argv[++i]);
        } else if (a == "--repetition-penalty" || a == "--rep-penalty") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(2); }
            opt.repetition_penalty = std::atof(argv[++i]);
        } else if (a == "--do-sample" || a == "--sample") {
            opt.do_sample = true;
        } else if (a == "--perf") {
            opt.enable_perf = true;
            if (opt.perf_level == 0) opt.perf_level = 1;
        } else if (a == "--perf-level") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --perf-level\n"; std::exit(2); }
            opt.perf_level = std::atoi(argv[++i]);
            opt.enable_perf = true;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return opt;
}
