#include "prompt.h"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <stdexcept>

static std::string lstrip_newlines(const std::string& s) {
    size_t start = s.find_first_not_of('\n');
    return (start == std::string::npos) ? "" : s.substr(start);
}

static std::string rstrip_newlines(const std::string& s) {
    size_t end = s.find_last_not_of('\n');
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

// ==========================================
// CHATML TEMPLATE (Qwen3 / MiniCPM4)
// ==========================================

static std::string apply_chatml_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt,
    bool enable_thinking,
    bool minicpm5,
    bool qwen35,
    bool qwen3
) {
    std::stringstream prompt;
    bool has_tools = !tools.empty();

    // System message handling
    if (has_tools) {
        prompt << "<|im_start|>system\n";
        if (qwen35) {
            prompt << "# Tools\n\nYou have access to the following functions:\n\n<tools>";
            for (const auto& tool : tools) {
                prompt << "\n" << tool.dump();
            }
            prompt << "\n</tools>\n\n"
                   << "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                   << "<tool_call>\n<function=example_function_name>\n"
                   << "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                   << "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
                   << "</function>\n</tool_call>\n\n"
                   << "<IMPORTANT>\nReminder:\n"
                   << "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
                   << "- Required parameters MUST be specified\n"
                   << "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
                   << "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
                   << "</IMPORTANT>";
            if (!messages.empty() && messages[0].role == "system") {
                std::string s_content = rstrip_newlines(lstrip_newlines(messages[0].content));
                if (!s_content.empty()) {
                    prompt << "\n\n" << s_content;
                }
            }
        } else if (minicpm5) {
            if (!messages.empty() && messages[0].role == "system") {
                prompt << messages[0].content << "\n\n";
            }
            prompt << "# Tools\n\n"
                   << "You are provided with function signatures within <tools></tools> XML tags:\n"
                   << "<tools>";
            for (const auto& tool : tools) {
                prompt << "\n" << tool.dump();
            }
            prompt << "\n</tools>\n\n"
                   << "Tool usage guidelines:\n"
                   << "- You may call zero or more functions. If no function calls are needed, just answer normally.\n"
                   << "- When calling a function, return an XML object using "
                   << "<function name=\"function-name\"><param name=\"param-name\">param-value</param></function>.\n"
                   << "- Include every required parameter and do not add text after the function call.\n";
        } else {
            if (!messages.empty() && messages[0].role == "system") {
                prompt << messages[0].content << "\n\n";
            }
            prompt << "# Tools\n\n"
                   << "You may call one or more functions to assist with the user query.\n\n"
                   << "You are provided with function signatures within <tools></tools> XML tags:\n"
                   << "<tools>";
            for (const auto& tool : tools) {
                prompt << "\n" << tool.dump();
            }
            prompt << "\n</tools>\n\n"
                   << "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                   << "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        }
        prompt << "<|im_end|>\n";
    } else {
        if (!messages.empty() && messages[0].role == "system") {
            prompt << "<|im_start|>system\n" << messages[0].content << "<|im_end|>\n";
        }
    }

    // Determine multi-step tool sequence
    bool multi_step_tool = true;
    int last_query_index = (int)messages.size() - 1;

    for (int i = (int)messages.size() - 1; i >= 0; --i) {
        const auto& msg = messages[i];
        bool is_tool_response = false;
        if (msg.content.size() >= 15) {
            if (msg.content.rfind("<tool_response>", 0) == 0 &&
                msg.content.find("</tool_response>") != std::string::npos) {
                is_tool_response = true;
            }
        }
        if (multi_step_tool && msg.role == "user" && !is_tool_response) {
            multi_step_tool = false;
            last_query_index = i;
        }
    }

    // Process messages
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        std::string content = msg.content;

        if (msg.role == "system" && i == 0) continue;

        if (msg.role == "user" || msg.role == "system") {
            prompt << "<|im_start|>" << msg.role << "\n" << content << "<|im_end|>\n";
        }
        else if (msg.role == "assistant") {
            std::string reasoning_content = msg.reasoning_content;
            std::string final_content = content;

            if (reasoning_content.empty() && final_content.find("</think>") != std::string::npos) {
                size_t start_think = final_content.find("<think>");
                size_t end_think = final_content.find("</think>");
                if (start_think != std::string::npos && end_think != std::string::npos) {
                    std::string extracted = final_content.substr(start_think + 7, end_think - (start_think + 7));
                    reasoning_content = lstrip_newlines(rstrip_newlines(extracted));
                    std::string remainder = final_content.substr(end_think + 8);
                    final_content = lstrip_newlines(remainder);
                }
            }

            bool is_after_last_query = ((int)i > last_query_index);
            bool is_last_message = (i == messages.size() - 1);
            bool has_reasoning = !reasoning_content.empty();

            bool show_thinking = enable_thinking && is_after_last_query && (is_last_message || has_reasoning);

            prompt << "<|im_start|>" << msg.role << "\n";

            if (show_thinking && has_reasoning) {
                prompt << "<think>\n" << reasoning_content << "\n</think>\n\n";
            }

            if (!final_content.empty()) {
                prompt << final_content;
            }

            if (!msg.tool_calls.empty()) {
                if (show_thinking || !final_content.empty()) prompt << "\n";
                for (size_t t = 0; t < msg.tool_calls.size(); ++t) {
                    if (t > 0) prompt << "\n";
                    json tc_obj = msg.tool_calls[t];
                    if (tc_obj.contains("function")) tc_obj = tc_obj["function"];
                    std::string fn_name = tc_obj.contains("name") ? tc_obj["name"].get<std::string>() : "";
                    json args = tc_obj.contains("arguments") ? tc_obj["arguments"] : json::object();
                    if (args.is_string()) {
                        auto parsed = json::parse(args.get<std::string>(), nullptr, false);
                        if (!parsed.is_discarded()) args = parsed;
                    }

                    if (minicpm5) {
                        prompt << "<function name=\"" << fn_name << "\">";
                        if (args.is_object()) {
                            for (auto it = args.begin(); it != args.end(); ++it) {
                                prompt << "<param name=\"" << it.key() << "\">"
                                       << (it.value().is_string() ? it.value().get<std::string>() : it.value().dump())
                                       << "</param>";
                            }
                        }
                        prompt << "</function>";
                    } else if (qwen35) {
                        prompt << "<tool_call>\n<function=" << fn_name << ">\n";
                        if (args.is_object()) {
                            for (auto it = args.begin(); it != args.end(); ++it) {
                                prompt << "<parameter=" << it.key() << ">\n"
                                       << (it.value().is_string() ? it.value().get<std::string>() : it.value().dump())
                                       << "\n</parameter>\n";
                            }
                        }
                        prompt << "</function>\n</tool_call>";
                    } else {
                        prompt << "<tool_call>\n"
                               << "{\"name\": \"" << fn_name << "\", "
                               << "\"arguments\": " << args.dump() << "}\n"
                               << "</tool_call>";
                    }
                }
            }
            prompt << "<|im_end|>\n";
        }
        else if (msg.role == "tool") {
            if (i == 0 || messages[i-1].role != "tool") prompt << "<|im_start|>user";
            prompt << "\n<tool_response>\n" << content << "\n</tool_response>";
            if (i == messages.size() - 1 || messages[i+1].role != "tool") prompt << "<|im_end|>\n";
        }
    }

    if (add_generation_prompt) {
        prompt << "<|im_start|>assistant\n";
        if (enable_thinking) {
            prompt << "<think>\n";
        
        }
    }

    return prompt.str();
}

// ==========================================
// YOUTU LLM TEMPLATE
// ==========================================

static std::string apply_youtu_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt
) {
    std::stringstream prompt;
    
    // Build system prompt
    std::string system_prompt;
    bool is_first_sp = true;
    
    // Collect system messages
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            if (is_first_sp) {
                system_prompt += msg.content;
                is_first_sp = false;
            } else {
                system_prompt += "\n\n" + msg.content;
            }
        }
    }
    
    // Check if there are tool messages
    bool has_tool_message = false;
    size_t first_tool_index = messages.size();
    
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        if (!has_tool_message && 
            (msg.role == "tool" || 
             (msg.role == "user" && msg.content.rfind("<tool_response>", 0) == 0 &&
              msg.content.find("</tool_response>") != std::string::npos))) {
            has_tool_message = true;
            first_tool_index = i;
        }
    }
    
    // Add tool descriptions if tools are provided
    if (!tools.empty()) {
        std::string tool_desc = "<|begin_of_tool_description|>Tool calling capabilities.\n"
                               "You may call one or more functions to assist with the user query. You have the following functions available:";
        
        for (const auto& tool : tools) {
            tool_desc += "\n```json\n" + tool.dump() + "\n```";
        }
        
        tool_desc += "\nFor tool call returns, you MUST use the following format:\n"
                    "<tool_call>{\"name\": \"function-name\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}</tool_call>\n"
                    "<|end_of_tool_description|>";
        
        if (system_prompt.empty()) {
            system_prompt = tool_desc;
        } else {
            system_prompt += "\n\n" + tool_desc;
        }
    }
    
    // Output: bos_token + system_prompt
    // Note: bos_token is typically handled by tokenizer, we just output system_prompt
    if (!system_prompt.empty()) {
        prompt << system_prompt;
    }
    
    // Process messages
    bool is_first = false;  // Not used in YouTu template like ChatML
    bool is_tool = false;
    bool is_last_user = false;
    
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        
        if (msg.role == "system") continue;  // Already handled above
        
        if (msg.role == "user") {
            is_tool = false;
            is_first = false;
            is_last_user = true;
            
            // Check if this is a tool response message
            if (msg.content.rfind("<tool_response>", 0) == 0 &&
                msg.content.find("</tool_response>") != std::string::npos) {
                // Tool response is already formatted correctly
                prompt << "<|User|>" << msg.content;
            } else {
                prompt << "<|User|>" << msg.content;
            }
        }
        else if (msg.role == "assistant") {
            is_last_user = false;
            std::string content = msg.content;
            
            // Handle <think> tags if present
            std::string think_content;
            size_t think_start = content.find("<think>");
            size_t think_end = content.find("</think>");
            
            if (think_start != std::string::npos && think_end != std::string::npos) {
                think_content = content.substr(think_start + 7, think_end - (think_start + 7));
                think_content = lstrip_newlines(rstrip_newlines(think_content));
                content = content.substr(0, think_start) + content.substr(think_end + 8);
                content = lstrip_newlines(content);
            }
            
            prompt << "<|Assistant|>";
            
            // Output think content if present
            if (!think_content.empty()) {
                prompt << "<think>" << think_content << "</think>";
            }
            
            // Output main content
            if (!content.empty()) {
                prompt << content;
            }
            
            // Output tool calls
            if (!msg.tool_calls.empty()) {
                for (const auto& tc : msg.tool_calls) {
                    json tc_obj = tc;
                    if (tc_obj.contains("function")) tc_obj = tc_obj["function"];
                    
                    prompt << "<tool_call>{\"name\": \"" 
                           << tc_obj["name"].get<std::string>() << "\", "
                           << "\"arguments\": " << tc_obj["arguments"].dump() 
                           << "}</tool_call>";
                }
            }
        }
        else if (msg.role == "tool") {
            // Tool messages are appended to the next user message or output directly
            if (i == 0 || messages[i-1].role != "tool") {
                prompt << "<|User|><tool_response>" << msg.content << "</tool_response>";
            } else {
                // Multiple tool messages in sequence
                prompt << msg.content;
            }
        }
    }
    
    if (add_generation_prompt) {
        prompt << "<|Assistant|>";
    }
    
    return prompt.str();
}


// MiniCPM-V-4.7's checkpoint template differs from Qwen3.5 even when
// thinking is disabled. Keep this restricted chat path independent of Qwen.
static std::string apply_minicpm_v47_template(const std::vector<Message>& messages,
    const std::vector<json>& tools, bool add_generation_prompt, bool enable_thinking) {
    if (messages.empty()) throw std::invalid_argument("MiniCPM requires messages");
    if (!tools.empty()) throw std::invalid_argument("MiniCPM experimental chat does not support tools");
    auto trim = [](const std::string& s) {
        const auto first=s.find_first_not_of(" \t\r\n\v\f");
        return first==std::string::npos ? std::string() : s.substr(first,s.find_last_not_of(" \t\r\n\v\f")-first+1);
    };
    int last_query=-1;
    for (size_t i=0;i<messages.size();++i) {
        const auto& m=messages[i];
        if (!m.tool_calls.empty() || (m.role!="system" && m.role!="user" && m.role!="assistant"))
            throw std::invalid_argument("MiniCPM experimental chat supports only system/user/assistant messages");
        if (m.role=="system" && i!=0) throw std::invalid_argument("System message must be first");
        const auto content=trim(m.content);
        if (m.role=="user" && !(content.rfind("<tool_response>",0)==0 &&
            content.size()>=16 && content.compare(content.size()-16,16,"</tool_response>")==0)) last_query=(int)i;
    }
    // CLI incremental prefill may render only the initial system fragment.
    if(last_query<0 && !(messages.size()==1 && messages[0].role=="system" && !add_generation_prompt))
        throw std::invalid_argument("MiniCPM requires a user query");
    std::ostringstream out;
    for(size_t i=0;i<messages.size();++i) {
        const auto& m=messages[i];
        auto content=trim(m.content), reasoning=trim(m.reasoning_content);
        out<<"<|im_start|>"<<m.role<<"\n";
        if(m.role=="assistant") {
            const auto close=content.find("</think>");
            if(m.reasoning_content.empty() && close!=std::string::npos) {
                auto before=content.substr(0,close);
                const auto open=before.rfind("<think>");
                reasoning=trim(open==std::string::npos?before:before.substr(open+7));
                content=lstrip_newlines(content.substr(content.rfind("</think>")+8));
            }
            out<<"<think>\n";
            if((int)i>last_query) out<<reasoning;
            out<<"\n</think>\n\n";
        }
        out<<content<<"<|im_end|>\n";
    }
    if(add_generation_prompt) {
        out<<"<|im_start|>assistant\n<think>\n";
        if(!enable_thinking) out<<"\n</think>\n\n";
    }
    return out.str();
}

// ==========================================
// PUBLIC API
// ==========================================

std::string apply_chat_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt,
    bool enable_thinking
) {
    return apply_chatml_template(messages, tools, add_generation_prompt, enable_thinking, false, false, false);
}

std::string apply_youtu_chat_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt
) {
    return apply_youtu_template(messages, tools, add_generation_prompt);
}

std::string apply_chat_template(
    TemplateType type,
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt,
    bool enable_thinking
) {
    switch (type) {
        case TemplateType::MINICPM_V47:
            return apply_minicpm_v47_template(messages, tools, add_generation_prompt, enable_thinking);
        case TemplateType::YOUTU:
            return apply_youtu_template(messages, tools, add_generation_prompt);
        case TemplateType::QWEN3:
            return apply_chatml_template(messages, tools, add_generation_prompt, enable_thinking, false, false, true);
        case TemplateType::MINICPM5:
            return apply_chatml_template(messages, tools, add_generation_prompt, enable_thinking, true, false, false);
        case TemplateType::QWEN35:
            return apply_chatml_template(messages, tools, add_generation_prompt, enable_thinking, false, true, false);
        case TemplateType::CHATML:
        default:
            return apply_chatml_template(messages, tools, add_generation_prompt, enable_thinking, false, false, false);
    }
}
