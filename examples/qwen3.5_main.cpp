// Copyright 2026 Tencent
// SPDX-License-Identifier: Apache-2.0

#include "qwen3.5_0.8b.h"
#include "utils/prompt.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "mat.h"
#include "net.h"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s [image-path]\n", argv[0]);
        return -1;
    }

    const char* image_path = argv[1];

    cv::Mat bgr = cv::imread(image_path, 1);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_path);
        return -1;
    }

    qwen3_5_0p8b model("./assets/qwen3.5_0.8b/qwen3.5_vision_embed_patch.ncnn.param",
                      "./assets/qwen3.5_0.8b/qwen3.5_vision_embed_patch.ncnn.bin",
                      "./assets/qwen3.5_0.8b/qwen3.5_vision_embed_pos.ncnn.param",
                      "./assets/qwen3.5_0.8b/qwen3.5_vision_embed_pos.ncnn.bin",
                      "./assets/qwen3.5_0.8b/qwen3.5_vision_encoder.ncnn.param",
                      "./assets/qwen3.5_0.8b/qwen3.5_vision_encoder.ncnn.bin",
                      "./assets/qwen3.5_0.8b/qwen3.5_embed_token.ncnn.param",
                      "./assets/qwen3.5_0.8b/qwen3.5_embed_token.ncnn.bin",
                      "./assets/qwen3.5_0.8b/qwen3.5_proj_out.ncnn.param",
                      "./assets/qwen3.5_0.8b/qwen3.5_decoder.ncnn.param",
                      "./assets/qwen3.5_0.8b/qwen3.5_decoder.ncnn.bin",
                      "./assets/qwen3.5_0.8b/vocab.txt",
                      "./assets/qwen3.5_0.8b/merges.txt",
                       /*use_vulkan=*/false);

    std::cout << "Chat with Qwen3.5-0.8B! Type 'exit' or 'quit' to end the conversation.\n";

    std::string prompt = apply_chat_template({
        {"system", "You are a helpful assistant."},
    }, {}, false, false);

    // std::cout << "prompt = " << prompt << std::endl;

    auto ctx = model.prefill(prompt);

    {
        // std::string user_input = "识别图片中的文字";
        std::string user_input = "分析图片内容";
        // std::string user_input = "Describe this image.";

        std::string user_message = apply_chat_template({
            {"user", "<|vision_start|><|image_pad|><|vision_end|>" + user_input}
        }, {}, true);

        // std::cout << "user_message = " << user_message << std::endl;

        ctx = model.prefill(user_message, bgr, ctx);

        GenerateConfig cfg;
        cfg.beam_size = 1;
        cfg.top_k = 20;
        cfg.top_p = 0.8;
        cfg.temperature = 0.7;
        cfg.repetition_penalty = 1.0;
        cfg.do_sample = true;
        cfg.max_new_tokens = 32768;

        ctx = model.generate(ctx, cfg, [](const std::string& token){
            std::cout << token << std::flush;
        });
        std::cout << std::endl;
    }

    while (true) {
        std::string input;
        std::cout << "User: ";
        std::getline(std::cin, input);
        if (input == "exit" || input == "quit") {
            break;
        }
        std::string user_message = apply_chat_template({
            {"user", input}
        }, {}, true);

        ctx = model.prefill(user_message, ctx);

        std::cout << "Assistant: ";

        GenerateConfig cfg;
        cfg.beam_size = 1;
        cfg.top_k = 20;
        cfg.top_p = 0.8;
        cfg.temperature = 0.7;
        cfg.repetition_penalty = 1.0;
        cfg.do_sample = true;
        cfg.max_new_tokens = 32768;

        ctx = model.generate(ctx, cfg, [](const std::string& token){
            std::cout << token << std::flush;
        });
        std::cout << std::endl;
    }

    return 0;
}
