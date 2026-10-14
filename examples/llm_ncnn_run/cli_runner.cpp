#include "cli_runner.h"

#include "json_utils.h"
#include "tools.h"

#include <iostream>
#include <fstream>

TemplateType detect_template_type(const std::string& model_path) {
    try {
        std::ifstream ifs(model_path + "/model.json");
        if (!ifs.is_open()) {
            return TemplateType::CHATML;
        }
        
        json config;
        ifs >> config;
        
        if (config.value("setting",json::object()).value("vision",json::object()).value("type",std::string())=="minicpm_v4_7")
            return TemplateType::MINICPM_V47;
        if (config.contains("type")) {
            std::string type = config["type"].get<std::string>();
            if (type == "youtu_llm") {
                return TemplateType::YOUTU;
            }
            if (type == "qwen3") {
                return TemplateType::QWEN3;
            }
            if (type == "minicpm5") {
                return TemplateType::MINICPM5;
            }
            if (type == "minicpm_v4_7" || type == "minicpmv4_7") return TemplateType::MINICPM_V47;
            if (type == "qwen3.5") {
                return TemplateType::QWEN35;
            }
        }
    } catch (...) {
    }
    
    return TemplateType::CHATML;
}

int run_cli(const Options& opt,
            ncnn_llm_gpt& model,
            const std::vector<json>& builtin_tools,
            const std::unordered_map<std::string, std::function<json(const json&)>>& builtin_router,
            TemplateType template_type,
            const ncnn::Mat& image) {
    std::cout << "llm_ncnn_run (cli). Type 'exit' or 'quit' to end the conversation.\n";
    const char* template_name = template_type == TemplateType::YOUTU ? "YouTu" :
                                template_type == TemplateType::QWEN3 ? "Qwen3" :
                                template_type == TemplateType::MINICPM5 ? "MiniCPM5" :
                                template_type == TemplateType::MINICPM_V47 ? "MiniCPM-V-4.7" :
                                template_type == TemplateType::QWEN35 ? "Qwen3.5" : "ChatML";
    std::cout << "Using template: " << template_name << "\n";

    const bool allow_tools = template_type != TemplateType::MINICPM_V47;
    if (!allow_tools && !builtin_tools.empty())
        std::cout << "MiniCPM-V-4.7 experimental chat: built-in tools disabled.\n";
    std::string system_prompt = "You are a helpful assistant.";
    std::shared_ptr<ncnn_llm_gpt_ctx> ctx;
    if (allow_tools && !builtin_tools.empty() && model.supports_tool_calling()) {
        ctx = model.define_tools(nullptr, builtin_tools, system_prompt, template_type);
    } else {
        std::string prompt = apply_chat_template(template_type, {{"system", system_prompt}}, {}, false, false);
        ctx = model.prefill(prompt);
    }

    bool has_image = !ncnn_mat_empty(image);
    bool first_turn = true;

    while (true) {
        std::string input;
        std::cout << "User: ";
        if (!std::getline(std::cin, input)) break;
        if (input == "exit" || input == "quit") break;

        if (first_turn && has_image) {
            std::string user_message = apply_chat_template(template_type, {
                {"user", model.image_prompt_placeholder() + input}
            }, {}, true, opt.enable_thinking);
            ctx = model.prefill(user_message, image, ctx);
            first_turn = false;
        } else {
            std::string user_message = apply_chat_template(template_type, {
                {"user", input}
            }, {}, true, opt.enable_thinking);
            ctx = model.prefill(user_message, ctx);
        }

        std::cout << "Assistant: ";
        GenerateConfig cfg;
        cfg.max_new_tokens = opt.max_new_tokens;
        cfg.top_k = opt.top_k;
        cfg.top_p = opt.top_p;
        cfg.temperature = opt.temperature;
        cfg.repetition_penalty = opt.repetition_penalty;
        cfg.do_sample = opt.do_sample;
        cfg.enable_thinking = opt.enable_thinking;
        cfg.enable_perf = opt.enable_perf;
        cfg.perf_level = opt.perf_level;

        if (allow_tools) cfg.tool_callback = [&](const json& call) {
            json result;
            try {
                std::string fname = call.at("name").get<std::string>();
                json args = call.value("arguments", json::object());
                bool handled = false;

                if (!builtin_tools.empty()) {
                    if (auto it = builtin_router.find(fname); it != builtin_router.end()) {
                        result = it->second(args);
                        handled = true;
                    }
                }

                if (!handled) {
                    result = json{{"error", "unknown function"}, {"name", fname}};
                }
            } catch (const std::exception& e) {
                result = json{{"error", e.what()}};
            }

            std::cout << "\n[Tool Call]: " << call.dump() << "\n";
            std::cout << "[Tool Result]: " << result.dump() << "\n";
            std::cout << "Assistant: " << std::flush;

            return json{
                {"result", result},
                {"call", call}
            };
        };

        ctx = model.generate(ctx, cfg, [](const std::string& token) {
            std::cout << token << std::flush;
        });

        std::cout << "\n";
    }

    return 0;
}
