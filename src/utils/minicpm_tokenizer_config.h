#pragma once
#include <filesystem>
#include <string>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace ncnn_llm {
// Resolve the opt-in artifact before loading weights. Model packages are local,
// but their configuration must not redirect artifact reads outside the package.
inline std::string minicpm_tokenizer_artifact(const std::string& model_dir,
    const nlohmann::json& tokenizer, bool is_minicpm) {
    if (!is_minicpm) {
        if (tokenizer.contains("pretokenizer"))
            throw std::runtime_error("Pretokenizer metadata is supported only for MiniCPM-V-4.7");
        return {};
    }
    if (!tokenizer.contains("pretokenizer"))
        throw std::runtime_error("MiniCPM requires its exact pretokenizer artifact; reconvert this model");
    const auto& spec=tokenizer.at("pretokenizer");
    if (spec.at("type").get<std::string>()!="minicpm_v47")
        throw std::runtime_error("Unsupported MiniCPM pretokenizer type");
    const auto name=spec.at("file").get<std::string>();
    if (name.empty() || name.find(':')!=std::string::npos || name.find('\\')!=std::string::npos)
        throw std::runtime_error("Invalid MiniCPM tokenizer artifact path");
    const std::filesystem::path relative(name);
    if (relative.is_absolute()) throw std::runtime_error("MiniCPM tokenizer artifact must be relative");
    auto path=std::filesystem::canonical(model_dir);
    for(const auto& component:relative) {
        if(component=="." || component=="..") throw std::runtime_error("MiniCPM tokenizer artifact traversal");
        path/=component;
        if(std::filesystem::is_symlink(path)) throw std::runtime_error("MiniCPM tokenizer artifact symlink");
    }
    if(!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path)>1024*1024)
        throw std::runtime_error("MiniCPM tokenizer artifact missing or exceeds 1 MiB");
    return path.string();
}
}
