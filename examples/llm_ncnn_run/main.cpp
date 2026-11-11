#include "cli_runner.h"
#include "options.h"
#include "tools.h"
#include "utf8_args.h"

#include "ncnn_llm_gpt.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string normalize_model_path(std::string path) {
    std::filesystem::path p(path);
    if (p.is_absolute()) return path;
    if (!p.has_parent_path()) {
        return (std::filesystem::path("./assets") / p).string();
    }
    return path;
}

}

int main(int argc, char** argv) {
    try {
        enable_utf8_console();
        std::vector<std::string> utf8_args = get_utf8_args(argc, argv);
        std::vector<char*> cargv;
        cargv.reserve(utf8_args.size());
        for (auto& s : utf8_args) cargv.push_back(const_cast<char*>(s.c_str()));

        Options opt = parse_options((int)cargv.size(), cargv.data());
        opt.model_path = normalize_model_path(opt.model_path);

        if (!std::filesystem::exists(opt.model_path)) {
            std::cerr << "Model path does not exist: " << opt.model_path << "\n";
            return 1;
        }

        TemplateType template_type = detect_template_type(opt.model_path);

        ncnn_llm_gpt model(opt.model_path, opt.use_vulkan, opt.num_threads, opt.vulkan_device, opt.use_bf16, opt.force_naive);
        std::vector<json> builtin_tools = opt.enable_builtin_tools ? make_builtin_tools() : std::vector<json>();
        auto builtin_router = make_builtin_router();

        ncnn::Mat image;
        if (!opt.image_path.empty()) {
            image = load_image_to_ncnn_mat(opt.image_path);
            if (ncnn_mat_empty(image)) {
                std::cerr << "Failed to load image: " << opt.image_path << "\n";
                return 1;
            }
            std::cerr << "Image loaded: " << opt.image_path << "\n";
        }
        return run_cli(opt, model, builtin_tools, builtin_router, template_type, image);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize or run model: " << e.what() << "\n";
        return 1;
    }
}
