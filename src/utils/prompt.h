#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Template type enum
enum class TemplateType {
    CHATML,     // Qwen3, MiniCPM4 style <|im_start|>...<|im_end|>
    QWEN3,      // Qwen3 ChatML with an explicit thinking block toggle
    MINICPM5,   // MiniCPM5 ChatML with an explicit empty thinking block
    QWEN35,     // Qwen3.5 official tool-call XML format
    YOUTU,      // YouTu LLM style <|User|>...<|Assistant|>
    MINICPM_V47 // Official MiniCPM-V-4.7 single-image/text chat (no tools)
};

struct Message {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::vector<json> tool_calls;

    Message() = default;
    Message(std::string r, std::string c, std::string rc = "", std::vector<json> tc = {})
        : role(std::move(r)), content(std::move(c)), reasoning_content(std::move(rc)), tool_calls(std::move(tc)) {}
};

// ChatML template (default)
std::string apply_chat_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools = {},
    bool add_generation_prompt = true,
    bool enable_thinking = true
);

// YouTu LLM template
std::string apply_youtu_chat_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools = {},
    bool add_generation_prompt = true
);

// Generic template dispatcher
std::string apply_chat_template(
    TemplateType type,
    const std::vector<Message>& messages,
    const std::vector<json>& tools = {},
    bool add_generation_prompt = true,
    bool enable_thinking = true
);
