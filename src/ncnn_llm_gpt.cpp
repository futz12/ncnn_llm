#include "ncnn_llm_gpt.h"
#include "sampling.h"

static std::shared_ptr<ncnn_llm_gpt_ctx> clone_ctx(const std::shared_ptr<ncnn_llm_gpt_ctx>& src) {
    return src->clone();
}

static std::shared_ptr<ncnn_llm_gpt_ctx> create_ctx(int sconv_cnt, int gdr_cnt) {
    if (sconv_cnt > 0 || gdr_cnt > 0) {
        return std::make_shared<qwen3_5_ctx>();
    }
    return std::make_shared<ncnn_llm_gpt_base_ctx>();
}

// Class Implementation

ncnn_llm_gpt::ncnn_llm_gpt(const std::string& model_path, bool use_vulkan, int num_threads, int vulkan_device) 
    : vision_type(Vision_Type::VISION_CLOSE) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            ifs >> config;
        }
        
        // Load base model
        decoder_net = std::make_shared<ncnn::Net>();
        embed_net = std::make_shared<ncnn::Net>();
        proj_out_net = std::make_shared<ncnn::Net>();

        // Set number of threads (0 = use ncnn default which is get_cpu_count())
        if (num_threads > 0) {
            decoder_net->opt.num_threads = num_threads;
            embed_net->opt.num_threads = num_threads;
            proj_out_net->opt.num_threads = num_threads;
        }

        if (use_vulkan) {
            printf("[ncnn_llm_gpt] Vulkan enabled, using device %d\n", vulkan_device >= 0 ? vulkan_device : 0);
            // Only decoder_net uses Vulkan for compute-intensive operations

            // Set specific Vulkan device BEFORE enabling vulkan compute
#if NCNN_VULKAN
            if (vulkan_device >= 0) {
                decoder_net->opt.vulkan_device_index = vulkan_device;
            }
#endif
            decoder_net->opt.use_bf16_storage = true;
            // decoder_net->opt.use_bf16_packed = true;
            decoder_net->opt.use_fp16_arithmetic = false;
            decoder_net->opt.use_fp16_storage = false;
            decoder_net->opt.use_fp16_packed = false;
            decoder_net->opt.use_vulkan_compute = true;
        } else {
            printf("[ncnn_llm_gpt] Vulkan disabled, using CPU only\n");
        }


        std::string decoder_param = model_path + "/" + config["params"]["decoder_param"].get<std::string>();
        std::string decoder_bin = model_path + "/" + config["params"]["decoder_bin"].get<std::string>();
        std::string embed_param = model_path + "/" + config["params"]["embed_token_param"].get<std::string>();
        std::string embed_bin = model_path + "/" + config["params"]["embed_token_bin"].get<std::string>();
        std::string proj_out_param = model_path + "/" + config["params"]["proj_out_param"].get<std::string>();
        std::string proj_out_bin = model_path + "/" + config["params"]["proj_out_bin"].get<std::string>();

        printf("Loading model from %s\n", model_path.c_str());
        printf("  decoder param: %s\n", decoder_param.c_str());
        printf("  decoder bin: %s\n", decoder_bin.c_str());
        printf("  embed param: %s\n", embed_param.c_str());
        printf("  embed bin: %s\n", embed_bin.c_str());
        printf("  proj_out param: %s\n", proj_out_param.c_str());
        printf("  proj_out bin: %s\n", proj_out_bin.c_str());

        register_gdr_layers(*decoder_net);

        decoder_net->load_param(decoder_param.c_str());
        decoder_net->load_model(decoder_bin.c_str());
        embed_net->load_param(embed_param.c_str());
        embed_net->load_model(embed_bin.c_str());
        proj_out_net->load_param(proj_out_param.c_str());
        proj_out_net->load_model(proj_out_bin.c_str());

        // Load tokenizer
        std::string type = "bpe";
        if (config["tokenizer"].contains("type")) {
            type = config["tokenizer"]["type"].get<std::string>();
        }
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

        bpe = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe"
        ));

        std::vector<std::string> additional_special_tokens = config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>();
        for (const auto& token : additional_special_tokens) {
            bpe->AddAdditionalSpecialToken(token);
        }

        auto eos_token = config["tokenizer"]["eos"].get<std::string>();
        eos = (eos_token != "") ? bpe->token_to_id().at(eos_token) : -1;

        auto bos_token = config["tokenizer"]["bos"].get<std::string>();
        bos = (bos_token != "") ? bpe->token_to_id().at(bos_token) : -1;

        // Model settings
        if (config["setting"].contains("attn_cnt")) {
            attn_cnt = config["setting"]["attn_cnt"].get<int>();
        }
        if (config["setting"].contains("sconv_cnt")) {
            sconv_cnt = config["setting"]["sconv_cnt"].get<int>();
        }
        if (config["setting"].contains("gdr_cnt")) {
            gdr_cnt = config["setting"]["gdr_cnt"].get<int>();
        }

        if (config["setting"].contains("rope")) {
            auto rope_cfg = config["setting"]["rope"];
            if (rope_cfg.contains("rope_head_dim")) {
                rope_head_dim = rope_cfg["rope_head_dim"].get<int>();
            }
            if (rope_cfg["type"] == "LongRoPE") {
                rope_type = RoPE_Type::LongRoPE;
                short_factor = rope_cfg["short_factor"].get<std::vector<float>>();
                long_factor = rope_cfg["long_factor"].get<std::vector<float>>();
                original_max_position_embeddings = rope_cfg["original_max_position_embeddings"].get<int>();
            } else if (rope_cfg["type"] == "RoPE") {
                rope_type = RoPE_Type::RoPE;
            } else if (rope_cfg["type"] == "NTKRoPE") {
                // rope_scaling
                rope_type = RoPE_Type::NTK_RoPE;
            } else if (rope_cfg["type"] == "YaRNRoPE") {
                rope_type = RoPE_Type::YARN_RoPE;
            }

            if (rope_cfg.contains("rope_scaling"))
            {
                ntk_scaling_params.alpha = rope_cfg["rope_scaling"]["alpha"].get<float>();
                ntk_scaling_params.beta_fast = rope_cfg["rope_scaling"]["beta_fast"].get<float>();
                ntk_scaling_params.beta_slow = rope_cfg["rope_scaling"]["beta_slow"].get<float>();
                ntk_scaling_params.factor = rope_cfg["rope_scaling"]["factor"].get<float>();
                ntk_scaling_params.mscale = rope_cfg["rope_scaling"]["mscale"].get<float>();
                ntk_scaling_params.mscale_all_dim = rope_cfg["rope_scaling"]["mscale_all_dim"].get<float>();
            }

            rope_theta = rope_cfg["rope_theta"].get<float>();
        }

        if (config["setting"].contains("functions")) {
            auto func_cfg = config["setting"]["functions"];
            if (func_cfg["type"].get<std::string>() == "tool_call") {
                if (func_cfg.contains("tool_call_id")) {
                    tool_call_id = bpe->token_to_id().at(func_cfg["tool_call_id"].get<std::string>());
                    fprintf(stderr, "  tool_call_id: %d\n", tool_call_id);
                }
                if (func_cfg.contains("tool_call_end_id")) {
                    tool_call_end_id = bpe->token_to_id().at(func_cfg["tool_call_end_id"].get<std::string>());
                    fprintf(stderr, "  tool_call_end_id: %d\n", tool_call_end_id);
                }
            }
        }

        // Load think tokens if present in tokenizer
        auto it_think = bpe->token_to_id().find("<think>");
        if (it_think != bpe->token_to_id().end()) {
            think_id = it_think->second;
            fprintf(stderr, "  think_id: %d\n", think_id);
        }
        auto it_think_end = bpe->token_to_id().find("</think>");
        if (it_think_end != bpe->token_to_id().end()) {
            think_end_id = it_think_end->second;
            fprintf(stderr, "  think_end_id: %d\n", think_end_id);
        }

        // Vision settings
        std::string vision_type_str = "close";
        if (config["setting"].contains("vision")) {
            auto vision_cfg = config["setting"]["vision"];
            vision_type_str = vision_cfg["type"].get<std::string>();
            
            if (vision_type_str != "close") {
                if (vision_type_str == "vit") {
                    vision_type = Vision_Type::VISION_VIT;
                } else if (vision_type_str == "qwen3.5_vl") {
                    vision_type = Vision_Type::VISION_QWEN3_5_VL;
                }
                
                std::string vision_embed_patch_param = model_path + "/" + vision_cfg["vision_embed_patch_param"].get<std::string>();
                std::string vision_embed_patch_bin = model_path + "/" + vision_cfg["vision_embed_patch_bin"].get<std::string>();
                std::string vision_encoder_param = model_path + "/" + vision_cfg["vision_encoder_param"].get<std::string>();
                std::string vision_encoder_bin = model_path + "/" + vision_cfg["vision_encoder_bin"].get<std::string>();

                fprintf(stderr, "  vision embed patch param: %s\n", vision_embed_patch_param.c_str());
                fprintf(stderr, "  vision embed patch bin: %s\n", vision_embed_patch_bin.c_str());
                fprintf(stderr, "  vision encoder param: %s\n", vision_encoder_param.c_str());
                fprintf(stderr, "  vision encoder bin: %s\n", vision_encoder_bin.c_str());

                vision_embed_patch = std::make_shared<ncnn::Net>();
                vision_encoder = std::make_shared<ncnn::Net>();

                if (use_vulkan) {
                    vision_embed_patch->opt.use_vulkan_compute = true;
                    vision_encoder->opt.use_vulkan_compute = true;
                }
                vision_embed_patch->load_param(vision_embed_patch_param.c_str());
                vision_embed_patch->load_model(vision_embed_patch_bin.c_str());
                vision_encoder->load_param(vision_encoder_param.c_str());
                vision_encoder->load_model(vision_encoder_bin.c_str());

                if (vision_cfg.contains("vision_embed_pos_param")) {
                    std::string vision_embed_pos_param = model_path + "/" + vision_cfg["vision_embed_pos_param"].get<std::string>();
                    std::string vision_embed_pos_bin = model_path + "/" + vision_cfg["vision_embed_pos_bin"].get<std::string>();
                    fprintf(stderr, "  vision embed pos param: %s\n", vision_embed_pos_param.c_str());
                    fprintf(stderr, "  vision embed pos bin: %s\n", vision_embed_pos_bin.c_str());
                    
                    vision_embed_pos = std::make_shared<ncnn::Net>();
                    if (use_vulkan) {
                        vision_embed_pos->opt.use_vulkan_compute = true;
                    }
                    vision_embed_pos->load_param(vision_embed_pos_param.c_str());
                    vision_embed_pos->load_model(vision_embed_pos_bin.c_str());
                }

                auto it = bpe->token_to_id().find("<|image_pad|>");
                if (it != bpe->token_to_id().end()) {
                    image_pad_id = it->second;
                }

                // Load vision config
                patch_size = vision_cfg["patch_size"].get<int>();
                patch_dim = vision_cfg["patch_dim"].get<int>();
                max_num_patches = vision_cfg["max_num_patches"].get<int>();
                spatial_merge_size = vision_cfg["spatial_merge_size"].get<int>();

                if (vision_cfg.contains("rope")) {
                    auto rope_cfg = vision_cfg["rope"];
                    if (rope_cfg["type"] == "mRoPE") {
                        vision_rope_type = VisionRoPE_Type::mRoPE;
                        mrope_section = rope_cfg["mrope_section"].get<std::vector<int>>();
                    }
                }
            }
        }
    } catch (std::exception &e) {
        throw std::runtime_error(std::string("ncnn_llm_gpt load model failed: ") + e.what());
    }
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text) const {
    auto token_ids = bpe->encode(input_text, false, false);
    if (bos >= 0) token_ids.insert(token_ids.begin(), bos);

    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache, sin_cache;
    if (rope_type == RoPE_Type::LongRoPE) {
        generate_rope_embed_cache_LongRoPE(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
    } else if (rope_type == RoPE_Type::NTK_RoPE) {
        generate_ntk_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::YARN_RoPE) {
        generate_yarn_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    }
    else
    {
        generate_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta);
    }

    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", input_ids_mat);
        ex.extract("out0", token_embed);
    }

    ncnn::Mat mask((int)token_ids.size(), (int)token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < (int)token_ids.size(); j++) {
            row[j] = -1e38f;
        }
    }

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;
    std::vector<ncnn::Mat> sconv_cache;
    std::vector<ncnn::Mat> gdr_cache;
    ncnn::Mat decode_out;
    {
        ncnn::Extractor ex = decoder_net->create_extractor();
        ex.input("in0", token_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            sconv_cache.emplace_back(std::move(cache));
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            gdr_cache.emplace_back(std::move(cache));
        }
    }

    // Handle last token
    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    
    ncnn::Mat last_cos_cache, last_sin_cache;
    if (rope_type == RoPE_Type::LongRoPE) {
        generate_rope_embed_cache_LongRoPE(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
    } else if (rope_type == RoPE_Type::NTK_RoPE) {
        generate_ntk_rope_embed_cache(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::YARN_RoPE) {
        generate_yarn_rope_embed_cache(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    }
    else {
        generate_rope_embed_cache(1, rope_head_dim, (int)token_ids.size(), last_cos_cache, last_sin_cache, rope_theta);
    }

    ncnn::Mat last_mask((int)token_ids.size() + 1, 1);
    last_mask.fill(0.0f);

    {
        ncnn::Extractor ex = decoder_net->create_extractor();
        ex.input("in0", last_token_embed);
        ex.input("in1", last_mask);
        ex.input("in2", last_cos_cache);
        ex.input("in3", last_sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_in[16], name_v_in[16];
            std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
            std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
            ex.input(name_k_in, kv_cache[i].first);
            ex.input(name_v_in, kv_cache[i].second);
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_in[16];
            std::snprintf(name_in, sizeof(name_in), "cache_conv%d", i);
            ex.input(name_in, sconv_cache[i]);
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_in[16];
            std::snprintf(name_in, sizeof(name_in), "cache_gdr%d", i);
            ex.input(name_in, gdr_cache[i]);
        }

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            sconv_cache[i] = std::move(cache);
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
            ncnn::Mat cache;
            ex.extract(name_out, cache);
            gdr_cache[i] = std::move(cache);
        }

        ex.extract("out0", decode_out);
    }

    ncnn::Mat logits;
    {
        ncnn::Extractor ex = proj_out_net->create_extractor();
        ex.input("in0", decode_out);
        ex.extract("out0", logits);
    }

    int next_token_id = 0;
    {
        const float* p = logits;
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                next_token_id = i;
            }
        }
    }

    auto ctx = create_ctx(sconv_cnt, gdr_cnt);
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = (int)token_ids.size() + 1;
    
    if (sconv_cnt > 0 || gdr_cnt > 0) {
        auto qwen_ctx = std::dynamic_pointer_cast<qwen3_5_ctx>(ctx);
        if (qwen_ctx) {
            qwen_ctx->sconv_cache = std::move(sconv_cache);
            qwen_ctx->gdr_cache = std::move(gdr_cache);
        }
    }
    
    return ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text, const ncnn::Mat& bgr, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const {
    std::shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);

    ncnn::Mat image_embeds;
    int num_patches_w = 0;
    int num_patches_h = 0;
    get_visiual_features(bgr, image_embeds, num_patches_w, num_patches_h);

    const int image_embeds_size = image_embeds.h;

    auto token_ids = bpe->encode(input_text, false, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", input_ids_mat);
        ex.extract("out0", token_embed);
    }

    int image_pad_index = -1;
    inject_image_embeds(token_ids, token_embed, image_pad_index, image_pad_id, image_embeds);

    ncnn::Mat cos_cache, sin_cache;
    if (image_embeds.empty()) {
        generate_rope_embed_cache(token_ids.size(), rope_head_dim, new_ctx->position_id, cos_cache, sin_cache, rope_theta);
        new_ctx->position_id += token_ids.size();
    } else {
        if (vision_type == Vision_Type::VISION_QWEN3_5_VL) {
            generate_rope_embed_cache_vision_mrope_interleaved(token_ids.size(), rope_head_dim, new_ctx->position_id, image_pad_index, image_embeds_size, num_patches_w, cos_cache, sin_cache, rope_theta);
        } else {
            generate_rope_embed_cache_vision_mrope(token_ids.size(), rope_head_dim, new_ctx->position_id, image_pad_index, image_embeds_size, num_patches_w, spatial_merge_size, mrope_section, cos_cache, sin_cache, rope_theta);
        }
        new_ctx->position_id += token_ids.size() - image_embeds_size + (num_patches_w / spatial_merge_size);
    }

    ncnn::Mat mask((int)token_ids.size() + new_ctx->kv_cache[0].first.h, (int)token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++) {
        float* row = mask.row(i);
        for (int j = new_ctx->kv_cache[0].first.h + i + 1; j < (int)token_ids.size() + new_ctx->kv_cache[0].first.h; j++) {
            row[j] = -1e38f;
        }
    }

    ncnn::Mat decode_out;
    {
        ncnn::Extractor ex = decoder_net->create_extractor();
        ex.input("in0", token_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, new_ctx->kv_cache[i].first);
            ex.input(vname, new_ctx->kv_cache[i].second);
        }

        auto qwen_ctx = std::dynamic_pointer_cast<qwen3_5_ctx>(new_ctx);
        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[16];
                std::snprintf(name, sizeof(name), "cache_conv%d", i);
                ex.input(name, qwen_ctx->sconv_cache[i]);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[16];
                std::snprintf(name, sizeof(name), "cache_gdr%d", i);
                ex.input(name, qwen_ctx->gdr_cache[i]);
            }
        }

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(kname, k_cache);
            ex.extract(vname, v_cache);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_conv%d", i);
                ncnn::Mat cache;
                ex.extract(name, cache);
                qwen_ctx->sconv_cache[i] = std::move(cache);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_gdr%d", i);
                ncnn::Mat cache;
                ex.extract(name, cache);
                qwen_ctx->gdr_cache[i] = std::move(cache);
            }
        }
    }

    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    
    ncnn::Mat last_cos_cache, last_sin_cache;
    generate_rope_embed_cache(1, rope_head_dim, new_ctx->position_id, last_cos_cache, last_sin_cache, rope_theta);
    new_ctx->position_id += 1;

    ncnn::Mat last_mask(new_ctx->kv_cache[0].first.h + 1, 1);
    last_mask.fill(0.0f);

    {
        ncnn::Extractor ex = decoder_net->create_extractor();
        ex.input("in0", last_token_embed);
        ex.input("in1", last_mask);
        ex.input("in2", last_cos_cache);
        ex.input("in3", last_sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char kname[16], vname[16];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, new_ctx->kv_cache[i].first);
            ex.input(vname, new_ctx->kv_cache[i].second);
        }

        auto qwen_ctx = std::dynamic_pointer_cast<qwen3_5_ctx>(new_ctx);
        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[16];
                std::snprintf(name, sizeof(name), "cache_conv%d", i);
                ex.input(name, qwen_ctx->sconv_cache[i]);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[16];
                std::snprintf(name, sizeof(name), "cache_gdr%d", i);
                ex.input(name, qwen_ctx->gdr_cache[i]);
            }
        }

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(kname, k_cache);
            ex.extract(vname, v_cache);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_conv%d", i);
                ncnn::Mat cache;
                ex.extract(name, cache);
                qwen_ctx->sconv_cache[i] = std::move(cache);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_gdr%d", i);
                ncnn::Mat cache;
                ex.extract(name, cache);
                qwen_ctx->gdr_cache[i] = std::move(cache);
            }
        }

        ex.extract("out0", decode_out);
    }

    ncnn::Mat logits;
    {
        ncnn::Extractor ex = proj_out_net->create_extractor();
        ex.input("in0", decode_out);
        ex.extract("out0", logits);
    }
    
    int next_token_id = 0;
    {
        const float* p = logits;
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                next_token_id = i;
            }
        }
    }
    new_ctx->cur_token = next_token_id;
    return new_ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const {
    std::shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);

    auto token_ids = bpe->encode(input_text, false, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    ncnn::Mat cos_cache, sin_cache;
    int current_pos = new_ctx->position_id;

    if (rope_type == RoPE_Type::LongRoPE) {
        generate_rope_embed_cache_LongRoPE(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
    } else if (rope_type == RoPE_Type::NTK_RoPE) {
        generate_ntk_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::YARN_RoPE) {
        generate_yarn_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
    }
    else {
        generate_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, cos_cache, sin_cache, rope_theta);
    }
    new_ctx->position_id += token_ids.size();
    
    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", input_ids_mat);
        ex.extract("out0", token_embed);
    }

    ncnn::Mat mask((int)token_ids.size() + new_ctx->kv_cache[0].first.h, (int)token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < (int)token_ids.size(); i++) {
        float* row = mask.row(i);
        for (int j = new_ctx->kv_cache[0].first.h + i + 1; j < (int)token_ids.size() + new_ctx->kv_cache[0].first.h; j++) {
            row[j] = -1e38f;
        }
    }
    
    ncnn::Mat decode_out;
    {
        ncnn::Extractor ex = decoder_net->create_extractor();
        ex.input("in0", token_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, new_ctx->kv_cache[i].first);
            ex.input(vname, new_ctx->kv_cache[i].second);
        }

        auto qwen_ctx = std::dynamic_pointer_cast<qwen3_5_ctx>(new_ctx);
        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[16];
                std::snprintf(name, sizeof(name), "cache_conv%d", i);
                ex.input(name, qwen_ctx->sconv_cache[i]);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[16];
                std::snprintf(name, sizeof(name), "cache_gdr%d", i);
                ex.input(name, qwen_ctx->gdr_cache[i]);
            }
        }

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(kname, k_cache);
            ex.extract(vname, v_cache);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_conv%d", i);
                ncnn::Mat cache;
                ex.extract(name, cache);
                qwen_ctx->sconv_cache[i] = std::move(cache);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_gdr%d", i);
                ncnn::Mat cache;
                ex.extract(name, cache);
                qwen_ctx->gdr_cache[i] = std::move(cache);
            }
        }
    }

    ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
    ncnn::Mat last_token_embed;
    {
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", last_token_mat);
        ex.extract("out0", last_token_embed);
    }
    
    ncnn::Mat last_cos_cache, last_sin_cache;

    if (rope_type == RoPE_Type::LongRoPE) {
        generate_rope_embed_cache_LongRoPE(1, rope_head_dim, new_ctx->position_id, last_cos_cache, last_sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
    } else if (rope_type == RoPE_Type::NTK_RoPE) {
        generate_ntk_rope_embed_cache(1, rope_head_dim, new_ctx->position_id, last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    } else if (rope_type == RoPE_Type::YARN_RoPE) {
        generate_yarn_rope_embed_cache(1, rope_head_dim, new_ctx->position_id, last_cos_cache, last_sin_cache, rope_theta, ntk_scaling_params);
    }
    else {
        generate_rope_embed_cache(1, rope_head_dim, new_ctx->position_id, last_cos_cache, last_sin_cache, rope_theta);
    }
    new_ctx->position_id += 1;
    
    ncnn::Mat last_mask(new_ctx->kv_cache[0].first.h + 1, 1);
    last_mask.fill(0.0f);

    {
        ncnn::Extractor ex = decoder_net->create_extractor();
        ex.input("in0", last_token_embed);
        ex.input("in1", last_mask);
        ex.input("in2", last_cos_cache);
        ex.input("in3", last_sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char kname[16], vname[16];
            std::snprintf(kname, sizeof(kname), "cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "cache_v%d", i);
            ex.input(kname, new_ctx->kv_cache[i].first);
            ex.input(vname, new_ctx->kv_cache[i].second);
        }

        auto qwen_ctx = std::dynamic_pointer_cast<qwen3_5_ctx>(new_ctx);
        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[16];
                std::snprintf(name, sizeof(name), "cache_conv%d", i);
                ex.input(name, qwen_ctx->sconv_cache[i]);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[16];
                std::snprintf(name, sizeof(name), "cache_gdr%d", i);
                ex.input(name, qwen_ctx->gdr_cache[i]);
            }
        }

        for (int i = 0; i < attn_cnt; i++) {
            char kname[32], vname[32];
            std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
            std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(kname, k_cache);
            ex.extract(vname, v_cache);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_conv%d", i);
                ncnn::Mat cache;
                ex.extract(name, cache);
                qwen_ctx->sconv_cache[i] = std::move(cache);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_gdr%d", i);
                ncnn::Mat cache;
                ex.extract(name, cache);
                qwen_ctx->gdr_cache[i] = std::move(cache);
            }
        }

        ex.extract("out0", decode_out);
    }

    ncnn::Mat logits;
    {
        ncnn::Extractor ex = proj_out_net->create_extractor();
        ex.input("in0", decode_out);
        ex.extract("out0", logits);
    }
    
    int next_token_id = 0;
    {
        const float* p = logits;
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                next_token_id = i;
            }
        }
    }
    new_ctx->cur_token = next_token_id;

    return new_ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::generate(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in, const GenerateConfig& cfg, std::function<void(const std::string&)> callback) const {
    const int vocab_size = bpe->vocab_size();

    auto handle_tool = [&](const std::string& tool_call_text, std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_ref) {
        nlohmann::json tool_call_json;
        try {
            tool_call_json = nlohmann::json::parse(tool_call_text);
        } catch (const std::exception& e) {
            tool_call_json = nlohmann::json::object();
        }

        nlohmann::json tool_resp;
        if (cfg.tool_callback) {
            tool_resp = cfg.tool_callback(tool_call_json);
        } else {
            tool_resp = nlohmann::json{{"tool_call", tool_call_json}};
        }

        std::string tool_response_pre = "<|im_end|>\n<|im_start|>user\n<tool_response>\n\n";
        std::string tool_response_post = "\n\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n";

        ctx_ref = prefill(tool_response_pre + tool_resp.dump() + tool_response_post, ctx_ref);
    };

    auto ctx = clone_ctx(ctx_in);
    std::unordered_set<int> history;
    history.insert(ctx->cur_token);

    bool flag_in_tool_call = false;
    std::string tool_call_content;

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        if (ctx->cur_token == eos) break;

        if (ctx->cur_token == tool_call_id) {
            flag_in_tool_call = true;
        } else if (ctx->cur_token == tool_call_end_id) {
            flag_in_tool_call = false;
            handle_tool(tool_call_content, ctx);
            tool_call_content.clear();
            history.clear();
            history.insert(ctx->cur_token);
            continue;
        } else if (flag_in_tool_call) {
            tool_call_content += bpe->decode({ctx->cur_token}, false);
        } else {
            callback(bpe->decode({ctx->cur_token}, false));
        }

        ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
        ncnn::Mat cur_embed;
        {
            ncnn::Extractor ex = embed_net->create_extractor();
            ex.input("in0", cur_token_mat);
            ex.extract("out0", cur_embed);
        }

        ncnn::Mat cos_cache, sin_cache;
        if (rope_type == RoPE_Type::LongRoPE) {
            generate_rope_embed_cache_LongRoPE(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta, short_factor.data(), long_factor.data(), original_max_position_embeddings);
        } else if (rope_type == RoPE_Type::NTK_RoPE) {
            generate_ntk_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
        } else if (rope_type == RoPE_Type::YARN_RoPE) {
            generate_yarn_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta, ntk_scaling_params);
        }
        else {
            generate_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos_cache, sin_cache, rope_theta);
        }
        
        ctx->position_id++;

        ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
        mask.fill(0.f);

        ncnn::Mat decode_out;
        {
            ncnn::Extractor ex = decoder_net->create_extractor();
            ex.input("in0", cur_embed);
            ex.input("in1", mask);
            ex.input("in2", cos_cache);
            ex.input("in3", sin_cache);

            for (int i = 0; i < attn_cnt; ++i) {
                char kname[16], vname[16];
                std::snprintf(kname, sizeof(kname), "cache_k%d", i);
                std::snprintf(vname, sizeof(vname), "cache_v%d", i);
                ex.input(kname, ctx->kv_cache[i].first);
                ex.input(vname, ctx->kv_cache[i].second);
            }

            auto qwen_ctx = std::dynamic_pointer_cast<qwen3_5_ctx>(ctx);
            if (qwen_ctx) {
                for (int i = 0; i < sconv_cnt; ++i) {
                    char name[16];
                    std::snprintf(name, sizeof(name), "cache_conv%d", i);
                    ex.input(name, qwen_ctx->sconv_cache[i]);
                }
                for (int i = 0; i < gdr_cnt; ++i) {
                    char name[16];
                    std::snprintf(name, sizeof(name), "cache_gdr%d", i);
                    ex.input(name, qwen_ctx->gdr_cache[i]);
                }
            }

            for (int i = 0; i < attn_cnt; ++i) {
                char kname[32], vname[32];
                std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                ncnn::Mat k_cache, v_cache;
                ex.extract(kname, k_cache);
                ex.extract(vname, v_cache);
                ctx->kv_cache[i] = { std::move(k_cache), std::move(v_cache) };
            }

            if (qwen_ctx) {
                for (int i = 0; i < sconv_cnt; ++i) {
                    char name[32];
                    std::snprintf(name, sizeof(name), "out_cache_conv%d", i);
                    ncnn::Mat cache;
                    ex.extract(name, cache);
                    qwen_ctx->sconv_cache[i] = std::move(cache);
                }
                for (int i = 0; i < gdr_cnt; ++i) {
                    char name[32];
                    std::snprintf(name, sizeof(name), "out_cache_gdr%d", i);
                    ncnn::Mat cache;
                    ex.extract(name, cache);
                    qwen_ctx->gdr_cache[i] = std::move(cache);
                }
            }

            ex.extract("out0", decode_out);
        }

        ncnn::Mat logits_mat;
        {
            ncnn::Extractor ex = proj_out_net->create_extractor();
            ex.input("in0", decode_out);
            ex.extract("out0", logits_mat);
        }

        std::vector<float> logits(vocab_size);
        memcpy(logits.data(), logits_mat.data, sizeof(float) * vocab_size);

        for (int t : history) {
            if (t >= vocab_size) continue;
            if (logits[t] < 0) logits[t] *= cfg.repetition_penalty;
            else logits[t] /= cfg.repetition_penalty;
        }

        softmax_vec(logits, cfg.temperature);
        if (cfg.top_k > 0) apply_top_k(logits, cfg.top_k);
        if (cfg.top_p < 1.0f) apply_top_p(logits, cfg.top_p);

        int next_id;
        if (cfg.do_sample == 1) {
            next_id = sample_from_probs(logits);
        } else {
            next_id = std::max_element(logits.begin(), logits.end()) - logits.begin();
        }

        ctx->cur_token = next_id;
        history.insert(next_id);
    }
    return ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::define_tools(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx, const std::vector<nlohmann::json>& tools, const std::string& system_prompt) {
    if (tool_call_id < 0 || tool_call_end_id < 0) return ctx;

    this->tools = tools;
    std::string tool_prompt = apply_chat_template({{"system", system_prompt}}, tools, false, false);

    if (ctx) return prefill(tool_prompt, ctx);
    return prefill(tool_prompt);
}

// Vision Helper Implementations

int ncnn_llm_gpt::get_scaled_image_size(float scale, int size, int effective_patch_size) const {
    float scaled_size_f = (float)size * scale;
    int scaled_size = (int)(std::ceil(scaled_size_f / (float)effective_patch_size) * effective_patch_size);
    return std::max(effective_patch_size, scaled_size);
}

void ncnn_llm_gpt::get_image_size_for_patches(int image_height, int image_width, int patch_size, int max_num_patches, int& target_height, int& target_width) const {
    float scale = 1.0f;
    int effective_patch_size = patch_size * 2;
    while (true) {
        target_height = get_scaled_image_size(scale, image_height, effective_patch_size);
        target_width = get_scaled_image_size(scale, image_width, effective_patch_size);
        long long num_patches = ((long long)target_height / patch_size) * ((long long)target_width / patch_size);
        if (num_patches > max_num_patches) {
            scale -= 0.02f;
        } else {
            break;
        }
    }
}

ncnn::Mat ncnn_llm_gpt::bgr_to_pixel_values(const ncnn::Mat& bgr) const {
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    if (vision_type == Vision_Type::VISION_QWEN3_5_VL) {
        image_mean[0] = 0.5f; image_mean[1] = 0.5f; image_mean[2] = 0.5f;
        image_std[0] = 0.5f; image_std[1] = 0.5f; image_std[2] = 0.5f;
    }

    int img_h = bgr.h;
    int img_w = bgr.w;

    int num_patches_h = (img_h + patch_size - 1) / patch_size;
    int num_patches_w = (img_w + patch_size - 1) / patch_size;
    int num_patches = num_patches_h * num_patches_w;

    int embed_dim = patch_size * patch_size * 3;
    ncnn::Mat pixel_values(embed_dim, num_patches);

    const unsigned char* bgr_data = (const unsigned char*)bgr.data;

    for (int p = 0; p < num_patches; p++) {
        int ph = p / num_patches_w;
        int pw = p % num_patches_w;
        int start_y = ph * patch_size;
        int start_x = pw * patch_size;

        float* out_ptr = pixel_values.row(p);
        float* ptr_r = out_ptr;
        float* ptr_g = out_ptr + patch_size * patch_size;
        float* ptr_b = out_ptr + patch_size * patch_size * 2;

        for (int y = 0; y < patch_size; y++) {
            const unsigned char* img_row_ptr = NULL;
            int cur_img_y = start_y + y;
            if (cur_img_y < img_h) {
                img_row_ptr = bgr_data + cur_img_y * img_w * 3;
            }

            for (int x = 0; x < patch_size; x++) {
                int cur_img_x = start_x + x;
                if (img_row_ptr && cur_img_x < img_w) {
                    const unsigned char* pixel = img_row_ptr + cur_img_x * 3;
                    if (vision_type == Vision_Type::VISION_QWEN3_5_VL) {
                        *ptr_r++ = (pixel[2] / 255.5f - image_mean[0]) / image_std[0];
                        *ptr_g++ = (pixel[1] / 255.5f - image_mean[1]) / image_std[1];
                        *ptr_b++ = (pixel[0] / 255.5f - image_mean[2]) / image_std[2];
                    } else {
                        *ptr_r++ = (pixel[2] / 255.f - image_mean[0]) / image_std[0];
                        *ptr_g++ = (pixel[1] / 255.f - image_mean[1]) / image_std[1];
                        *ptr_b++ = (pixel[0] / 255.f - image_mean[2]) / image_std[2];
                    }
                } else {
                    float pad_val = 0.0f;
                    *ptr_r++ = pad_val;
                    *ptr_g++ = pad_val;
                    *ptr_b++ = pad_val;
                }
            }
        }
    }
    return pixel_values;
}

ncnn::Mat ncnn_llm_gpt::reorder_patches_for_merge(const ncnn::Mat& pixel_values, int h_patches, int w_patches, int merge_size) const {
    int num_patches = pixel_values.h;
    int feature_dim = pixel_values.w;

    if (num_patches != h_patches * w_patches) return ncnn::Mat();

    int grid_h = h_patches / merge_size;
    int grid_w = w_patches / merge_size;

    ncnn::Mat reordered_pixel_values(feature_dim, num_patches, (size_t)4u);
    int new_row_idx = 0;

    for (int gh = 0; gh < grid_h; gh++) {
        for (int gw = 0; gw < grid_w; gw++) {
            for (int mh = 0; mh < merge_size; mh++) {
                for (int mw = 0; mw < merge_size; mw++) {
                    int original_h = gh * merge_size + mh;
                    int original_w = gw * merge_size + mw;
                    int original_row_idx = original_h * w_patches + original_w;

                    const float* src_ptr = pixel_values.row(original_row_idx);
                    float* dst_ptr = reordered_pixel_values.row(new_row_idx);
                    memcpy(dst_ptr, src_ptr, feature_dim * sizeof(float));
                    new_row_idx++;
                }
            }
        }
    }
    return reordered_pixel_values;
}

void ncnn_llm_gpt::get_window_index(int num_patches_w, int num_patches_h, std::vector<int>& window_index, std::vector<int>& cu_window_seqlens) const {
    const int vit_merger_window_size = 4;

    int llm_grid_h = num_patches_h / spatial_merge_size;
    int llm_grid_w = num_patches_w / spatial_merge_size;

    int num_windows_h = (llm_grid_h + vit_merger_window_size - 1) / vit_merger_window_size;
    int num_windows_w = (llm_grid_w + vit_merger_window_size - 1) / vit_merger_window_size;

    window_index.clear();
    window_index.reserve(llm_grid_h * llm_grid_w);

    cu_window_seqlens.clear();
    cu_window_seqlens.push_back(0);

    int current_cu_len = 0;

    for (int nh = 0; nh < num_windows_h; ++nh) {
        for (int nw = 0; nw < num_windows_w; ++nw) {
            int h_start = nh * vit_merger_window_size;
            int w_start = nw * vit_merger_window_size;
            int h_end = std::min(h_start + vit_merger_window_size, llm_grid_h);
            int w_end = std::min(w_start + vit_merger_window_size, llm_grid_w);

            int valid_h = h_end - h_start;
            int valid_w = w_end - w_start;

            if (valid_h <= 0 || valid_w <= 0) continue;

            for (int r = h_start; r < h_end; ++r) {
                for (int c = w_start; c < w_end; ++c) {
                    int original_idx = r * llm_grid_w + c;
                    window_index.push_back(original_idx);
                }
            }

            int tokens_in_this_window = valid_h * valid_w * (spatial_merge_size * spatial_merge_size);
            current_cu_len += tokens_in_this_window;
            cu_window_seqlens.push_back(current_cu_len);
        }
    }
}

std::vector<float> ncnn_llm_gpt::compute_inv_freq(int dim, float theta) {
    int num_freqs = dim / 2;
    std::vector<float> inv_freq(num_freqs);
    for (int i = 0; i < num_freqs; i++) {
        float exp_val = (float)(i * 2) / (float)dim;
        inv_freq[i] = 1.0f / std::pow(theta, exp_val);
    }
    return inv_freq;
}

void ncnn_llm_gpt::generate_rope_embeds(int num_patches_w, int num_patches_h, ncnn::Mat& emb_cos, ncnn::Mat& emb_sin, int rope_dim) const {
    int seq_len = num_patches_w * num_patches_h;
    std::vector<float> inv_freq = compute_inv_freq(rope_dim);
    int half_dim = inv_freq.size();
    int output_dim = half_dim * 4;

    emb_cos.create(output_dim, seq_len, sizeof(float));
    emb_sin.create(output_dim, seq_len, sizeof(float));

    int idx = 0;
    int grid_h = num_patches_h / spatial_merge_size;
    int grid_w = num_patches_w / spatial_merge_size;

    for (int gh = 0; gh < grid_h; gh++) {
        for (int gw = 0; gw < grid_w; gw++) {
            for (int bi = 0; bi < spatial_merge_size; bi++) {
                for (int bj = 0; bj < spatial_merge_size; bj++) {
                    int current_h = gh * spatial_merge_size + bi;
                    int current_w = gw * spatial_merge_size + bj;

                    float* cos_ptr = emb_cos.row(idx);
                    float* sin_ptr = emb_sin.row(idx);

                    for (int k = 0; k < half_dim; k++) {
                        float freq = inv_freq[k];
                        float angle_h = (float)current_h * freq;
                        float angle_w = (float)current_w * freq;

                        int idx_h1 = k;
                        int idx_w1 = half_dim + k;
                        int idx_h2 = 2 * half_dim + k;
                        int idx_w2 = 3 * half_dim + k;

                        float ch = std::cos(angle_h);
                        float sh = std::sin(angle_h);
                        float cw = std::cos(angle_w);
                        float sw = std::sin(angle_w);

                        cos_ptr[idx_h1] = ch; sin_ptr[idx_h1] = sh;
                        cos_ptr[idx_h2] = ch; sin_ptr[idx_h2] = sh;
                        cos_ptr[idx_w1] = cw; sin_ptr[idx_w1] = sw;
                        cos_ptr[idx_w2] = cw; sin_ptr[idx_w2] = sw;
                    }
                    idx++;
                }
            }
        }
    }
}

int ncnn_llm_gpt::get_visiual_features(const ncnn::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h) const {
    if (ncnn_mat_empty(bgr)) {
        image_embeds.release();
        num_patches_w = 0;
        num_patches_h = 0;
        return 0;
    }

    int img_w = bgr.w;
    int img_h = bgr.h;

    int target_w, target_h;
    get_image_size_for_patches(img_h, img_w, patch_size, max_num_patches, target_h, target_w);

    ncnn::Mat bgr_resized = ncnn_mat_resize(bgr, target_w, target_h);

    num_patches_w = (target_w + patch_size - 1) / patch_size;
    num_patches_h = (target_h + patch_size - 1) / patch_size;
    const int seq_len = num_patches_w * num_patches_h;

    ncnn::Mat pixel_values = bgr_to_pixel_values(bgr_resized);
    pixel_values = reorder_patches_for_merge(pixel_values, num_patches_h, num_patches_w, spatial_merge_size);
    pixel_values = pixel_values.reshape(patch_size * patch_size, 1, 3, seq_len);

    {
        ncnn::Mat tmp(patch_size * patch_size, 2, 3, seq_len);
        for (int i = 0; i < seq_len; i++) {
            for (int c = 0; c < 3; c++) {
                const float* src = pixel_values.channel(i).depth(c).row(0);
                memcpy(tmp.channel(i).depth(c).row(0), src, patch_size * patch_size * sizeof(float));
                memcpy(tmp.channel(i).depth(c).row(1), src, patch_size * patch_size * sizeof(float));
            }
        }
        pixel_values = tmp.reshape(patch_size * patch_size * 2 * 3, seq_len);
    }

    std::vector<int> window_index;
    std::vector<int> cu_window_seqlens;
    get_window_index(num_patches_w, num_patches_h, window_index, cu_window_seqlens);

    ncnn::Mat patch_embeds(patch_dim, seq_len);
    for (int i = 0; i < seq_len; i++) {
        ncnn::Mat patch = pixel_values.row_range(i, 1).reshape(patch_size, patch_size, 2, 3);
        ncnn::Mat patch_embed;
        ncnn::Extractor ex = vision_embed_patch->create_extractor();
        ex.input("in0", patch);
        ex.extract("out0", patch_embed);
        memcpy(patch_embeds.row(i), patch_embed.reshape(patch_dim), patch_dim * sizeof(float));
    }

    if (vision_embed_pos) {
        ncnn::Mat pos_embeds;
        {
            ncnn::Mat grid(num_patches_w, num_patches_h);
            ncnn::Extractor ex = vision_embed_pos->create_extractor();
            ex.input("in0", grid);
            ex.extract("out0", pos_embeds);
        }
        
        pos_embeds = reorder_patches_for_merge(pos_embeds, num_patches_h, num_patches_w, spatial_merge_size);

        ncnn::Mat emb_cos, emb_sin;
        generate_rope_embeds(num_patches_w, num_patches_h, emb_cos, emb_sin, 32);

        {
            ncnn::Extractor ex = vision_encoder->create_extractor();
            ex.input("in0", patch_embeds);
            ex.input("in1", pos_embeds);
            ex.input("in2", emb_cos);
            ex.input("in3", emb_sin);
            ex.extract("out0", image_embeds);
        }
        return 0;
    }

    ncnn::Mat emb_cos, emb_sin;
    generate_rope_embeds(num_patches_w, num_patches_h, emb_cos, emb_sin, 40);

    ncnn::Mat patch_embeds_reordered(patch_embeds.w, seq_len, sizeof(float));
    ncnn::Mat emb_cos_reordered(emb_cos.w, seq_len, sizeof(float));
    ncnn::Mat emb_sin_reordered(emb_sin.w, seq_len, sizeof(float));

    int group_size = 4;
    for (int i = 0; i < window_index.size(); i++) {
        int src_group_idx = window_index[i];
        for (int k = 0; k < group_size; k++) {
            int src_row = src_group_idx * group_size + k;
            int dst_row = i * group_size + k;

            const float* src_ptr = patch_embeds.row(src_row);
            float* dst_ptr = patch_embeds_reordered.row(dst_row);
            memcpy(dst_ptr, src_ptr, patch_embeds.w * sizeof(float));

            const float* src_cos = emb_cos.row(src_row);
            float* dst_cos = emb_cos_reordered.row(dst_row);
            memcpy(dst_cos, src_cos, emb_cos.w * sizeof(float));

            const float* src_sin = emb_sin.row(src_row);
            float* dst_sin = emb_sin_reordered.row(dst_row);
            memcpy(dst_sin, src_sin, emb_sin.w * sizeof(float));
        }
    }

    std::vector<int> cu_seqlens = cu_window_seqlens;
    ncnn::Mat attention_mask(seq_len, seq_len);
    attention_mask.fill(-1e9f);

    for (size_t i = 1; i < cu_seqlens.size(); i++) {
        int start = cu_seqlens[i-1];
        int end = cu_seqlens[i];
        for (int r = start; r < end; r++) {
            float* row_ptr = attention_mask.row(r);
            for (int c = start; c < end; c++) {
                row_ptr[c] = 0.f;
            }
        }
    }

    {
        ncnn::Extractor ex = vision_encoder->create_extractor();
        ex.input("in0", patch_embeds_reordered);
        ex.input("in1", emb_cos_reordered);
        ex.input("in2", emb_sin_reordered);
        ex.input("in3", attention_mask);
        ex.extract("out0", image_embeds);
    }

    ncnn::Mat image_embeds_restored(image_embeds.w, image_embeds.h);
    for (int i = 0; i < window_index.size(); i++) {
        int dest_group_idx = window_index[i];
        int src_group_idx = i;
        const float* src_ptr = image_embeds.row(dest_group_idx);
        float* dst_ptr = image_embeds_restored.row(src_group_idx);
        memcpy(dst_ptr, src_ptr, image_embeds.w * sizeof(float));
    }
    image_embeds = image_embeds_restored;
    return 0;
}
