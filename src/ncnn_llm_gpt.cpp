#include <set>
#include <filesystem>
#include <cpu.h>
#include <limits>
#include "utils/perf_profiler.h"
#include "kernel/lm_head.h"
#include "embed.h"
#include "ncnn_llm_gpt.h"
#include "ncnn_text_runtime.h"
#include "utils/vision_rope.h"
#include "utils/minicpm_tokenizer_config.h"


// Shared loading and diagnostics for language, legacy vision and MiniCPM graphs.
static void load_net(ncnn::Net& net, const std::string& name,
                     const std::string& param_path, const std::string& bin_path) {
    const int param_ret = net.load_param(param_path.c_str());
    if (param_ret != 0) {
        throw std::runtime_error(name + " load_param failed (ret=" +
                                 std::to_string(param_ret) + "): " + param_path);
    }
    const int model_ret = net.load_model(bin_path.c_str());
    if (model_ret != 0) {
        throw std::runtime_error(name + " load_model failed (ret=" +
                                 std::to_string(model_ret) + "): " + bin_path);
    }
}

static nlohmann::json parse_tool_call_payload(const std::string& payload) {
    try {
        const std::string trimmed = payload;
        auto parsed = nlohmann::json::parse(trimmed, nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) return parsed;

        const size_t fn = payload.find("<function=");
        if (fn == std::string::npos) {
            // MiniCPM5 emits <function name=...><param name=...>...</param>.
            const size_t name_tag = payload.find("name=\"");
            if (name_tag == std::string::npos) return nlohmann::json::object();
            const size_t name_begin = name_tag + 6;
            const size_t name_end = payload.find('"', name_begin);
            if (name_end == std::string::npos) return nlohmann::json::object();

            nlohmann::json args = nlohmann::json::object();
            size_t cursor = payload.find('>', name_end);
            if (cursor == std::string::npos) return nlohmann::json::object();
            ++cursor;
            while (true) {
                const size_t tag = payload.find("<param", cursor);
                if (tag == std::string::npos) break;
                const size_t key_tag = payload.find("name=\"", tag);
                if (key_tag == std::string::npos) break;
                const size_t key_begin = key_tag + 6;
                const size_t key_end = payload.find('"', key_begin);
                if (key_end == std::string::npos) break;
                const size_t value_begin = payload.find('>', key_end);
                if (value_begin == std::string::npos) break;
                const size_t value_end = payload.find("</param>", value_begin + 1);
                if (value_end == std::string::npos) break;

                std::string value = payload.substr(value_begin + 1, value_end - value_begin - 1);
                const size_t first = value.find_first_not_of(" \t\r\n");
                const size_t last = value.find_last_not_of(" \t\r\n");
                value = first == std::string::npos ? std::string() : value.substr(first, last - first + 1);
                if (value.rfind("<![CDATA[", 0) == 0 && value.size() >= 12 &&
                    value.compare(value.size() - 3, 3, "]]>") == 0) {
                    value = value.substr(9, value.size() - 12);
                }
                auto value_json = nlohmann::json::parse(value, nullptr, false);
                args[payload.substr(key_begin, key_end - key_begin)] =
                    value_json.is_discarded() ? nlohmann::json(value) : value_json;
                cursor = value_end + 8;
            }

            return nlohmann::json{{"name", payload.substr(name_begin, name_end - name_begin)},
                                  {"arguments", std::move(args)}};
        }
        const size_t name_begin = fn + 10;
        const size_t name_end = payload.find('>', name_begin);
        if (name_end == std::string::npos) return nlohmann::json::object();

        nlohmann::json args = nlohmann::json::object();
        size_t cursor = name_end + 1;
        while (true) {
            const size_t tag = payload.find("<parameter=", cursor);
            if (tag == std::string::npos) break;
            const size_t key_begin = tag + 11;
            const size_t key_end = payload.find('>', key_begin);
            if (key_end == std::string::npos) break;
            const size_t value_end = payload.find("</parameter>", key_end + 1);
            if (value_end == std::string::npos) break;
            std::string value = payload.substr(key_end + 1, value_end - key_end - 1);
            const size_t first = value.find_first_not_of(" \t\r\n");
            const size_t last = value.find_last_not_of(" \t\r\n");
            value = first == std::string::npos ? std::string() : value.substr(first, last - first + 1);
            auto value_json = nlohmann::json::parse(value, nullptr, false);
            args[payload.substr(key_begin, key_end - key_begin)] =
                value_json.is_discarded() ? nlohmann::json(value) : value_json;
            cursor = value_end + 12;
        }

        return nlohmann::json{{"name", payload.substr(name_begin, name_end - name_begin)},
                              {"arguments", std::move(args)}};
    } catch (...) {
        return nlohmann::json::object();
    }
}

static std::shared_ptr<ncnn_llm_gpt_ctx> clone_ctx(const std::shared_ptr<ncnn_llm_gpt_ctx>& src) {
    if (!src) {
        throw std::runtime_error("LLM context must not be null");
    }
    return src->clone();
}

static std::shared_ptr<ncnn_llm_gpt_ctx> create_ctx(int sconv_cnt, int gdr_cnt) {
    std::shared_ptr<ncnn_llm_gpt_ctx> ctx;
    if (sconv_cnt > 0 || gdr_cnt > 0) {
        ctx = std::make_shared<qwen3_5_ctx>();
    } else {
        ctx = std::make_shared<ncnn_llm_gpt_base_ctx>();
    }
    ctx->kvcache_allocator = std::make_shared<ncnn::UnlockedPoolAllocator>();
    ctx->kvcache_allocator->set_size_compare_ratio(0.f);
    return ctx;
}

static void require_extract(int ret, const ncnn::Mat& output, const std::string& name) {
    if (ret != 0 || output.empty()) {
        throw std::runtime_error("ncnn extract failed for " + name + " (ret=" +
                                 std::to_string(ret) + ")");
    }
}



// Class Implementation

ncnn_llm_gpt::ncnn_llm_gpt(const std::string& model_path, bool use_vulkan, int num_threads, int vulkan_device, bool use_bf16, bool force_naive)
    : vision_type(Vision_Type::VISION_CLOSE) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            ifs >> config;
        }
        
        const bool minicpm_fp32 = config.value("setting", json::object()).value("vision", json::object()).value("type", std::string()) == "minicpm_v4_7";
        const auto minicpm_tokenizer_file = ncnn_llm::minicpm_tokenizer_artifact(model_path, config.at("tokenizer"), minicpm_fp32);
        if (minicpm_fp32) {
            if(!config.at("setting").contains("rope") || config.at("setting").at("rope").value("type",std::string())!="RoPE")
                throw std::runtime_error("MiniCPM requires an explicit unscaled RoPE configuration");
            fprintf(stderr, "[ncnn_llm_gpt] MiniCPM-V-4.7 experimental contract: forcing CPU FP32 for text and vision; Vulkan/BF16/FP16 disabled.\n");
            use_vulkan = false;
            use_bf16 = false;
        }
        model_path_ = model_path;
        use_vulkan_ = use_vulkan;
        {
            int default_threads = ncnn::get_physical_cpu_count();
            if (default_threads <= 0) default_threads = ncnn::get_cpu_count();
            if (default_threads <= 0) default_threads = 4;
            num_threads_ = num_threads > 0 ? num_threads : default_threads;
        }
        vulkan_device_ = vulkan_device;
        use_bf16_ = use_bf16;

        // Load base model
        decoder_net = std::make_shared<ncnn::Net>();
        embed_net = std::make_shared<ncnn::Net>();
        proj_out_net = std::make_shared<ncnn::Net>();
        lm_head = std::make_shared<ncnn_llm::LlmHead>();

        // Set number of threads across all nets
        decoder_net->opt.num_threads = num_threads_;
        embed_net->opt.num_threads = num_threads_;
        proj_out_net->opt.num_threads = num_threads_;

        if (use_vulkan) {
            printf("[ncnn_llm_gpt] Vulkan enabled, using device %d\n", vulkan_device >= 0 ? vulkan_device : 0);
            // Only decoder_net uses Vulkan for compute-intensive operations

            // Set specific Vulkan device BEFORE enabling vulkan compute
            if (vulkan_device >= 0) {
                decoder_net->opt.vulkan_device_index = vulkan_device;
            }
            decoder_net->opt.use_bf16_storage = true;
            // decoder_net->opt.use_bf16_packed = true;
            decoder_net->opt.use_fp16_arithmetic = false;
            decoder_net->opt.use_fp16_storage = false;
            decoder_net->opt.use_fp16_packed = false;
            decoder_net->opt.use_vulkan_compute = true;
        } else {
            printf("[ncnn_llm_gpt] Vulkan disabled, using CPU only\n");
#if defined(__riscv)
            // riscv CPUs have no fp16 coverage for the decomposed gated-norm chains
            // used by qwen3.5 hybrid models (pow->mean->rsqrt->mul on ~1e-2 magnitude activations):
            // fp16 storage collapses precision and greedy decoding degenerates to token 0.
            // Scope default fp32 to qwen3.5 to preserve SpaceMiT IME2 (smt.vfwmadot)
            // acceleration on qwen3; allow NCNN_RISCV_FP16=1 or NCNN_RISCV_FP32=1 to override.
            const std::string model_type = config.value("type", "");
            const char* fp16env = getenv("NCNN_RISCV_FP16");
            const char* fp32env = getenv("NCNN_RISCV_FP32");
            bool force_fp32 = (fp32env && fp32env[0] == '1');
            bool force_fp16 = (fp16env && fp16env[0] == '1');
            bool need_fp32 = force_fp32 || (!force_fp16 && model_type == "qwen3.5");
            if (need_fp32) {
                decoder_net->opt.use_fp16_storage = false;
                decoder_net->opt.use_fp16_arithmetic = false;
                decoder_net->opt.use_fp16_packed = false;
                embed_net->opt.use_fp16_storage = false;
                embed_net->opt.use_fp16_arithmetic = false;
                embed_net->opt.use_fp16_packed = false;
                proj_out_net->opt.use_fp16_storage = false;
                proj_out_net->opt.use_fp16_arithmetic = false;
                proj_out_net->opt.use_fp16_packed = false;
                printf("[ncnn_llm_gpt] riscv CPU: fp32 path enabled for %s (set NCNN_RISCV_FP16=1 to override)\n",
                       model_type.c_str());
            }
#endif
        }

        if (use_bf16) {
            decoder_net->opt.use_bf16_storage = true;
            printf("[ncnn_llm_gpt] BF16 storage enabled for decoder\n");
        }


        if (minicpm_fp32) {
            for (auto* net : {decoder_net.get(), embed_net.get(), proj_out_net.get()}) {
                net->opt.use_vulkan_compute = false;
                net->opt.use_bf16_storage = false;
                net->opt.use_fp16_storage = false;
                net->opt.use_fp16_arithmetic = false;
                net->opt.use_fp16_packed = false;
            }
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



        register_gdr_layers(*decoder_net, force_naive);

        const int decoder_param_ret = decoder_net->load_param(decoder_param.c_str());
        if (decoder_param_ret != 0) {
            throw std::runtime_error("decoder load_param failed (ret=" +
                                     std::to_string(decoder_param_ret) + "): " + decoder_param);
        }
        if (use_vulkan) {
            for (const auto* layer : decoder_net->layers()) {
                if (layer && !layer->support_vulkan) {
                    printf("[ncnn_llm_gpt] Notice: decoder contains layers not supported by Vulkan (such as block-quantized INT8 Gemm). Switching decoder to CPU.\n");
                    decoder_net->opt.use_vulkan_compute = false;
                    break;
                }
            }
        }
        const int decoder_model_ret = decoder_net->load_model(decoder_bin.c_str());
        if (decoder_model_ret != 0) {
            throw std::runtime_error("decoder load_model failed (ret=" +
                                     std::to_string(decoder_model_ret) + "): " + decoder_bin);
        }
        load_net(*embed_net, "embed", embed_param, embed_bin);

        bool tied_embeddings = (embed_bin == proj_out_bin);
        bool shared_success = false;

        if (tied_embeddings) {
            ncnn::Mat embed_weight;
            for (const auto* layer : embed_net->layers()) {
                if (layer && layer->type == "Embed") {
                    const auto* el = static_cast<const ncnn::Embed*>(layer);
                    embed_weight = el->weight_data;
                    break;
                }
            }

            if (!embed_weight.empty()) {
                ncnn::Option lm_opt = embed_net->opt;
                int ret = lm_head->init_shared(proj_out_param, embed_weight, lm_opt);
                if (ret == 0) {
                    shared_success = true;
                    printf("[ncnn_llm_gpt] Shared LM head with embedding weights (skipped loading %s, saved %zu MB)\n",
                           proj_out_bin.c_str(), (size_t)(embed_weight.total() * embed_weight.elemsize / (1024 * 1024)));
                } else {
                    printf("[ncnn_llm_gpt] Warning: Failed to initialize shared LM head (ret=%d), falling back to file load\n", ret);
                }
            }
        }

        if (!shared_success) {
            ncnn::Option lm_opt = embed_net->opt;
            int ret = lm_head->init_from_file(proj_out_param, proj_out_bin, lm_opt);
            if (ret != 0) {
                throw std::runtime_error("lm_head init_from_file failed (ret=" +
                                         std::to_string(ret) + "): " + proj_out_param);
            }
        }
        proj_out_net = lm_head->get_net();

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

        if (minicpm_fp32) bpe->LoadMiniCPMTokenizer(minicpm_tokenizer_file);

        auto eos_token = config["tokenizer"]["eos"].get<std::string>();
        eos = (eos_token != "") ? bpe->token_to_id().at(eos_token) : -1;
        eos_ids.clear();
        if (eos >= 0) eos_ids.insert(eos);
        if (config["tokenizer"].contains("eos_ids")) {
            for (const auto& value : config["tokenizer"]["eos_ids"]) {
                eos_ids.insert(value.get<int>());
            }
        }

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

        // Experimental MiniCPM image/text chat does not expose native tool dispatch.
        if (!minicpm_fp32 && config["setting"].contains("functions")) {
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
            
            if (vision_type_str == "minicpm_v4_7") {
                if(attn_cnt<=0 || attn_cnt>64 || sconv_cnt<=0 || sconv_cnt>192 || sconv_cnt!=gdr_cnt)
                    throw std::runtime_error("MiniCPM requires bounded full-attention and matched convolution/GDR cache groups");
                auto sections=vision_cfg.at("rope").at("mrope_section").get<std::vector<int>>();
                if(sections.size()!=3 || sections[0]<0 || sections[1]<0 || sections[2]<0 ||
                   int64_t(sections[0])+sections[1]+sections[2]!=rope_head_dim/2 ||
                   rope_head_dim<=0 || rope_head_dim%2 || !std::isfinite(rope_theta) || rope_theta<=0 || rope_type!=RoPE_Type::RoPE)
                    throw std::runtime_error("MiniCPM requires valid unscaled interleaved MRoPE parameters");
                for(const char* key:{"image","image_start","image_end","slice_start","slice_end","newline"}) {
                    const int id=vision_cfg.at("marker_ids").at(key);
                    if(id<0 || static_cast<size_t>(id)>=bpe->vocab_size())
                        throw std::runtime_error("MiniCPM marker id is outside the tokenizer vocabulary");
                }
                minicpm_vision = load_minicpm_vision(model_path, vision_cfg, num_threads);
                vision_type = Vision_Type::VISION_MINICPM_V4_7;
                image_pad_id = minicpm_vision->markers.image;
                mrope_section = vision_cfg.at("rope").at("mrope_section").get<std::vector<int>>();
                if (mrope_section.size() != 3) throw std::runtime_error("MiniCPM requires three MRoPE sections");
                fprintf(stderr, "  MiniCPM-V-4.7 experimental single-thumbnail vision (CPU FP32)\n");
            } else if (vision_type_str != "close") {
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
                load_net(*vision_embed_patch, "vision_embed_patch", vision_embed_patch_param, vision_embed_patch_bin);
                load_net(*vision_encoder, "vision_encoder", vision_encoder_param, vision_encoder_bin);

                if (vision_cfg.contains("vision_embed_pos_param")) {
                    std::string vision_embed_pos_param = model_path + "/" + vision_cfg["vision_embed_pos_param"].get<std::string>();
                    std::string vision_embed_pos_bin = model_path + "/" + vision_cfg["vision_embed_pos_bin"].get<std::string>();
                    fprintf(stderr, "  vision embed pos param: %s\n", vision_embed_pos_param.c_str());
                    fprintf(stderr, "  vision embed pos bin: %s\n", vision_embed_pos_bin.c_str());
                    
                    vision_embed_pos = std::make_shared<ncnn::Net>();
                    if (use_vulkan) {
                        vision_embed_pos->opt.use_vulkan_compute = true;
                    }
                    load_net(*vision_embed_pos, "vision_embed_pos", vision_embed_pos_param, vision_embed_pos_bin);
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

                if (vision_cfg.contains("rope_section")) {
                    vision_rope_section = vision_cfg["rope_section"].get<std::vector<int>>();
                } else if (vision_cfg.contains("head_dim")) {
                    int hd = vision_cfg["head_dim"].get<int>();
                    vision_rope_section = {hd / 4, hd / 4};
                } else {
                    int hd = (patch_dim == 1152) ? 72 : 64;
                    vision_rope_section = {hd / 4, hd / 4};
                }
            }
        }
    } catch (std::exception &e) {
        throw std::runtime_error(std::string("ncnn_llm_gpt load model failed: ") + e.what());
    }
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text) const {
    const auto t_prefill_start = std::chrono::steady_clock::now();

    auto ctx = create_ctx(sconv_cnt, gdr_cnt);
    ncnn_llm::PrefillPerfStats& pperf = ctx->prefill_perf;

    std::vector<int> token_ids;
    {
        ncnn_llm::ScopedTimer t(pperf.tokenizer_encode);
        token_ids = bpe->encode(input_text, false, false);
        if (bos >= 0 && !minicpm_vision) token_ids.insert(token_ids.begin(), bos);
    }

    const int total_prompt_tokens = (int)token_ids.size();
    if (token_ids.empty()) {
        throw std::runtime_error("input text produced no tokens");
    }

    ncnn::Mat cos_cache, sin_cache;
    {
        ncnn_llm::ScopedTimer t(pperf.rope_cache);
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
    }

    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn_llm::ScopedTimer t(pperf.embed_lookup);
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", input_ids_mat);
        require_extract(ex.extract("out0", token_embed), token_embed, "embed/out0");
    }

    ncnn::Mat mask((int)token_ids.size(), (int)token_ids.size());
    {
        ncnn_llm::ScopedTimer t(pperf.causal_mask);
        mask.fill(0.0f);
        for (int i = 0; i < (int)token_ids.size(); i++) {
            float* row = mask.row(i);
            for (int j = i + 1; j < (int)token_ids.size(); j++) {
                row[j] = -1e38f;
            }
        }
    }

    ncnn::Allocator* kv_alloc = ctx->kvcache_allocator ? ctx->kvcache_allocator.get() : nullptr;
    const int max_seqlen_hint = (int)token_ids.size() + 512;

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> kv_cache;
    std::vector<ncnn::Mat> sconv_cache;
    std::vector<ncnn::Mat> gdr_cache;
    ncnn::Mat decode_out;
    {
        ncnn_llm::ScopedTimer t(pperf.decoder_prompt);
        ncnn::Extractor ex = decoder_net->create_extractor();
        if (kv_alloc) {
            ex.set_kvcache_allocator(kv_alloc);
            ex.set_kvcache_max_seqlen_hint(max_seqlen_hint);
        }
        ex.input("in0", token_embed);
        ex.input("in1", mask);
        ex.input("in2", cos_cache);
        ex.input("in3", sin_cache);

        for (int i = 0; i < attn_cnt; i++) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            const int k_ret = ex.extract(name_k_out, k_cache, 1);
            const int v_ret = ex.extract(name_v_out, v_cache, 1);
            require_extract(k_ret, k_cache, name_k_out);
            require_extract(v_ret, v_cache, name_v_out);
            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        }

        for (int i = 0; i < sconv_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_conv%d", i);
            ncnn::Mat cache;
            const int ret = ex.extract(name_out, cache);
            require_extract(ret, cache, name_out);
            sconv_cache.emplace_back(std::move(cache));
        }

        for (int i = 0; i < gdr_cnt; i++) {
            char name_out[32];
            std::snprintf(name_out, sizeof(name_out), "out_cache_gdr%d", i);
            ncnn::Mat cache;
            const int ret = ex.extract(name_out, cache);
            require_extract(ret, cache, name_out);
            gdr_cache.emplace_back(std::move(cache));
        }
        require_extract(ex.extract("out0", decode_out), decode_out, "decoder/out0");
    }
        ncnn::Mat logits;
    {
        ncnn_llm::ScopedTimer t(pperf.lm_head);
        ncnn::Mat last_hidden = decode_out.row_range(total_prompt_tokens - 1, 1);
        logits = lm_head->forward(last_hidden, embed_net->opt);
    }

    int next_token_id = 0;
    {
        ncnn_llm::ScopedTimer t(pperf.sampling);
        const float* p = logits;
        float max_val = p[0];
        for (int i = 1; i < logits.w; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                next_token_id = i;
            }
        }
    }

    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = total_prompt_tokens;
    
    if (sconv_cnt > 0 || gdr_cnt > 0) {
        auto qwen_ctx = std::dynamic_pointer_cast<qwen3_5_ctx>(ctx);
        if (qwen_ctx) {
            qwen_ctx->sconv_cache = std::move(sconv_cache);
            qwen_ctx->gdr_cache = std::move(gdr_cache);
        }
    }

    const auto t_prefill_end = std::chrono::steady_clock::now();
    pperf.total_ms += std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    pperf.prompt_tokens += total_prompt_tokens;

    return ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text, const ncnn::Mat& bgr, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const {
    if (ncnn_mat_empty(bgr)) {
        throw std::runtime_error("image input is empty");
    }
    if (vision_type == Vision_Type::VISION_CLOSE || (!minicpm_vision && (!vision_embed_patch || !vision_encoder))) {
        throw std::runtime_error("model does not support image input");
    }

    if (minicpm_vision) {
        const size_t placeholder=input_text.find("<image>");
        if (placeholder==std::string::npos || input_text.find("<image>",placeholder+7)!=std::string::npos)
            throw std::runtime_error("MiniCPM image input requires exactly one <image> placeholder");
        if (ctx && ctx->image_count!=0) throw std::runtime_error("MiniCPM supports only one image per context; start a new context for another image");
    }
    const auto t_prefill_start = std::chrono::steady_clock::now();
    std::shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);
    ncnn_llm::PrefillPerfStats& pperf = new_ctx->prefill_perf;
    ncnn::Allocator* kv_alloc = new_ctx->kvcache_allocator ? new_ctx->kvcache_allocator.get() : nullptr;


    ncnn::Mat image_embeds;
    int num_patches_w = 0;
    int num_patches_h = 0;
    int vision_ret = 0;
    {
        ncnn_llm::ScopedTimer t(pperf.vision_feature_extract);
        vision_ret = get_visiual_features(bgr, image_embeds, num_patches_w, num_patches_h);
    }
    if (vision_ret != 0 || image_embeds.empty()) {
        throw std::runtime_error("vision feature extraction failed");
    }
    const int image_embeds_size = image_embeds.h;

    std::vector<int> token_ids;
    {
        ncnn_llm::ScopedTimer t(pperf.tokenizer_encode);
        if (minicpm_vision) {
            const size_t placeholder = input_text.find("<image>");
            if (placeholder == std::string::npos || input_text.find("<image>", placeholder + 7) != std::string::npos)
                throw std::runtime_error("MiniCPM image input requires exactly one <image> placeholder");
            token_ids = bpe->encode(input_text.substr(0, placeholder), false, false);
            token_ids.push_back(minicpm_vision->markers.image_start);
            token_ids.insert(token_ids.end(), image_embeds_size, image_pad_id);
            token_ids.push_back(minicpm_vision->markers.image_end);
            auto suffix = bpe->encode(input_text.substr(placeholder + 7), false, false);
            token_ids.insert(token_ids.end(), suffix.begin(), suffix.end());
        } else token_ids = bpe->encode(input_text, false, false);
    }
    int total_prompt_tokens = (int)token_ids.size();
    if (token_ids.empty()) {
        throw std::runtime_error("input text produced no tokens");
    }

    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    {
        ncnn_llm::ScopedTimer t(pperf.embed_lookup);
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", input_ids_mat);
        require_extract(ex.extract("out0", token_embed), token_embed, "embed/out0");
    }

    int image_pad_index = -1;
    inject_image_embeds(token_ids, token_embed, image_pad_index, image_pad_id,
                        image_embeds, static_cast<bool>(minicpm_vision));
    total_prompt_tokens = (int)token_ids.size();
    if (image_pad_index < 0) {
        throw std::runtime_error("image placeholder token was not found in the prompt");
    }

    ncnn::Mat cos_cache, sin_cache;
    if (minicpm_vision) {
        const auto positions = ncnn_llm::minicpm::canvas_positions(token_ids, minicpm_vision->markers,
            num_patches_h, num_patches_w, minicpm_vision->mode, new_ctx->position_id);
        std::vector<float> cc, ss;
        ncnn_llm::minicpm::interleaved_rope(positions, rope_head_dim, rope_theta,
            {mrope_section[0],mrope_section[1],mrope_section[2]}, cc, ss);
        cos_cache = ncnn::Mat(rope_head_dim/2, total_prompt_tokens, cc.data()).clone();
        sin_cache = ncnn::Mat(rope_head_dim/2, total_prompt_tokens, ss.data()).clone();
        if (positions.next_position > std::numeric_limits<int>::max()) throw std::runtime_error("MiniCPM position overflow");
        new_ctx->position_id = (int)positions.next_position;
    } else if (image_embeds.empty()) {
        generate_rope_embed_cache(token_ids.size(), rope_head_dim, new_ctx->position_id, cos_cache, sin_cache, rope_theta);
        new_ctx->position_id += token_ids.size();
    } else {
        if (vision_type == Vision_Type::VISION_QWEN3_5_VL) {
            generate_rope_embed_cache_vision_mrope_interleaved(token_ids.size(), rope_head_dim, new_ctx->position_id, image_pad_index, image_embeds_size, num_patches_w, num_patches_h, cos_cache, sin_cache, rope_theta);
        } else {
            generate_rope_embed_cache_vision_mrope(token_ids.size(), rope_head_dim, new_ctx->position_id, image_pad_index, image_embeds_size, num_patches_w, spatial_merge_size, mrope_section, cos_cache, sin_cache, rope_theta);
        }
        new_ctx->position_id += token_ids.size() - image_embeds_size +
                                (std::max(num_patches_w, num_patches_h) / spatial_merge_size);
    }

    const int max_seqlen_hint = (int)token_ids.size() + new_ctx->kv_cache[0].first.h + 512;
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
        ncnn_llm::ScopedTimer t(pperf.decoder_prompt);
        ncnn::Extractor ex = decoder_net->create_extractor();
        if (kv_alloc) {
            ex.set_kvcache_allocator(kv_alloc);
            ex.set_kvcache_max_seqlen_hint(max_seqlen_hint);
        }
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
            new_ctx->kv_cache[i].first.release();
            new_ctx->kv_cache[i].second.release();
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
            require_extract(ex.extract(kname, k_cache, 1), k_cache, kname);
            require_extract(ex.extract(vname, v_cache, 1), v_cache, vname);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_conv%d", i);
                ncnn::Mat cache;
                require_extract(ex.extract(name, cache), cache, name);
                qwen_ctx->sconv_cache[i] = std::move(cache);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_gdr%d", i);
                ncnn::Mat cache;
                require_extract(ex.extract(name, cache), cache, name);
                qwen_ctx->gdr_cache[i] = std::move(cache);
            }
        }
        require_extract(ex.extract("out0", decode_out), decode_out, "decoder/out0");
    }
        ncnn::Mat logits;
    {
        ncnn_llm::ScopedTimer t(pperf.lm_head);
        ncnn::Mat last_hidden = decode_out.row_range(total_prompt_tokens - 1, 1);
        logits = lm_head->forward(last_hidden, embed_net->opt);
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
    if (minicpm_vision) new_ctx->image_count=1;
    const auto t_prefill_end = std::chrono::steady_clock::now();
    pperf.total_ms += std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    pperf.prompt_tokens += total_prompt_tokens;
    return new_ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(const std::string& input_text, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const {
    const auto t_prefill_start = std::chrono::steady_clock::now();
    std::shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);
    ncnn_llm::PrefillPerfStats& pperf = new_ctx->prefill_perf;
    ncnn::Allocator* kv_alloc = new_ctx->kvcache_allocator ? new_ctx->kvcache_allocator.get() : nullptr;
    // Canvas MRoPE positions can be shorter than the physical image-token KV cache.
    const int cache_prefix = minicpm_vision ? new_ctx->kv_cache[0].first.h : new_ctx->position_id;
    const int max_seqlen_hint = cache_prefix + (int)input_text.size() + 512;

    std::vector<int> token_ids;
    {
        ncnn_llm::ScopedTimer t(pperf.tokenizer_encode);
        token_ids = bpe->encode(input_text, false, false);
    }
    const int total_prompt_tokens = (int)token_ids.size();
    if (token_ids.empty()) {
        throw std::runtime_error("input text produced no tokens");
    }

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
        ncnn_llm::ScopedTimer t(pperf.embed_lookup);
        ncnn::Extractor ex = embed_net->create_extractor();
        ex.input("in0", input_ids_mat);
        require_extract(ex.extract("out0", token_embed), token_embed, "embed/out0");
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
        ncnn_llm::ScopedTimer t(pperf.decoder_prompt);
        ncnn::Extractor ex = decoder_net->create_extractor();
        if (kv_alloc) {
            ex.set_kvcache_allocator(kv_alloc);
            ex.set_kvcache_max_seqlen_hint(max_seqlen_hint);
        }
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
            new_ctx->kv_cache[i].first.release();
            new_ctx->kv_cache[i].second.release();
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
            require_extract(ex.extract(kname, k_cache, 1), k_cache, kname);
            require_extract(ex.extract(vname, v_cache, 1), v_cache, vname);
            new_ctx->kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }

        if (qwen_ctx) {
            for (int i = 0; i < sconv_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_conv%d", i);
                ncnn::Mat cache;
                require_extract(ex.extract(name, cache), cache, name);
                qwen_ctx->sconv_cache[i] = std::move(cache);
            }
            for (int i = 0; i < gdr_cnt; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "out_cache_gdr%d", i);
                ncnn::Mat cache;
                require_extract(ex.extract(name, cache), cache, name);
                qwen_ctx->gdr_cache[i] = std::move(cache);
            }
        }
        require_extract(ex.extract("out0", decode_out), decode_out, "decoder/out0");
    }
        ncnn::Mat logits;
    {
        ncnn_llm::ScopedTimer t(pperf.lm_head);
        ncnn::Mat last_hidden = decode_out.row_range(total_prompt_tokens - 1, 1);
        logits = lm_head->forward(last_hidden, embed_net->opt);
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
    const auto t_prefill_end = std::chrono::steady_clock::now();
    pperf.total_ms += std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    pperf.prompt_tokens += total_prompt_tokens;

    return new_ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::generate(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in, const GenerateConfig& cfg, std::function<void(const std::string&)> callback) const {
    const int vocab_size = bpe->vocab_size();

    int perf_lvl = 0;
    if (cfg.perf_level > 0) {
        perf_lvl = cfg.perf_level;
    } else if (cfg.perf_level < 0) {
        perf_lvl = 0;
    } else if (cfg.enable_perf) {
        perf_lvl = 1;
    } else {
        perf_lvl = ncnn_llm::get_perf_level();
    }
    bool do_perf = (perf_lvl > 0);

    if (perf_lvl >= 2) {
#if NCNN_BENCHMARK
        ncnn::reset_layer_benchmark();
        ncnn::set_layer_benchmark_active(true);
#endif
    }

    ncnn_llm::DecodePerfStats decode_perf;
    const auto t_decode_start = std::chrono::steady_clock::now();

    auto handle_tool = [&](const std::string& tool_call_text, std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_ref) {
        nlohmann::json tool_call_json = parse_tool_call_payload(tool_call_text);

        nlohmann::json tool_resp;
        if (cfg.tool_callback) {
            tool_resp = cfg.tool_callback(tool_call_json);
        } else {
            tool_resp = nlohmann::json{{"tool_call", tool_call_json}};
        }

        std::string tool_response_pre = "<|im_end|>\n<|im_start|>user\n<tool_response>\n\n";
        std::string tool_response_post = cfg.enable_thinking
            ? "\n\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
            : "\n\n</tool_response><|im_end|>\n<|im_start|>assistant\n";

        ctx_ref = prefill(tool_response_pre + tool_resp.dump() + tool_response_post, ctx_ref);
    };

    auto ctx = clone_ctx(ctx_in);
    std::unordered_map<int, int> history_counts;
    history_counts[ctx->cur_token]++;
    std::vector<int> generated_tokens;
    generated_tokens.push_back(ctx->cur_token);

    bool flag_in_tool_call = false;
    bool flag_in_thinking = false;
    std::string tool_call_content;

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        if (eos_ids.count(ctx->cur_token)) break;

        if (ctx->cur_token == tool_call_id) {
            flag_in_tool_call = true;
            flag_in_thinking = false;
        } else if (ctx->cur_token == tool_call_end_id) {
            flag_in_tool_call = false;
            flag_in_thinking = false;
            handle_tool(tool_call_content, ctx);
            tool_call_content.clear();
            history_counts.clear();
            history_counts[ctx->cur_token]++;
            generated_tokens.clear();
            generated_tokens.push_back(ctx->cur_token);
            continue;
        } else if (flag_in_tool_call) {
            tool_call_content += bpe->decode({ctx->cur_token}, false);
        } else {
            if (ctx->cur_token == think_id) {
                flag_in_thinking = true;
            } else if (ctx->cur_token == think_end_id) {
                flag_in_thinking = false;
            }
            if (callback && (cfg.enable_thinking ||
                             (!flag_in_thinking && ctx->cur_token != think_id &&
                              ctx->cur_token != think_end_id))) {
                std::string token_str;
                {
                    ncnn_llm::ScopedTimer t(do_perf ? &decode_perf.tokenizer_decode : nullptr);
                    token_str = bpe->decode({ctx->cur_token}, false);
                }
                {
                    ncnn_llm::ScopedTimer t(do_perf ? &decode_perf.callback : nullptr);
                    callback(token_str);
                }
            }
        }

        ncnn::Mat cur_embed;
        {
            ncnn_llm::ScopedTimer t(do_perf ? &decode_perf.embed_lookup : nullptr);
            cur_embed = llm_run_text_embed(*embed_net, ctx->cur_token);
        }

        ncnn::Mat cos_cache, sin_cache;
        {
            ncnn_llm::ScopedTimer t(do_perf ? &decode_perf.rope_gen : nullptr);
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
        }
        
        ctx->position_id++;

        ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
        {
            ncnn_llm::ScopedTimer t(do_perf ? &decode_perf.mask_gen : nullptr);
            mask.fill(0.f);
        }

        const int decode_max_seqlen_hint =
            (minicpm_vision ? ctx->kv_cache[0].first.h + 1 : ctx->position_id) + cfg.max_new_tokens;
        ncnn::Mat decode_out;
        {
            ncnn_llm::ScopedTimer t(do_perf ? &decode_perf.decoder_step : nullptr);
            auto qwen_ctx = std::dynamic_pointer_cast<qwen3_5_ctx>(ctx);
            if (!qwen_ctx) {
                decode_out = llm_run_decoder_with_kv(*decoder_net, cur_embed, mask, cos_cache, sin_cache,
                                                     ctx->kv_cache, attn_cnt, false,
                                                     ctx->kvcache_allocator ? ctx->kvcache_allocator.get() : nullptr,
                                                     decode_max_seqlen_hint);
            } else {
                ncnn::Extractor ex = decoder_net->create_extractor();
                if (ctx->kvcache_allocator) {
                    ex.set_kvcache_allocator(ctx->kvcache_allocator.get());
                    ex.set_kvcache_max_seqlen_hint(decode_max_seqlen_hint);
                }
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
                    ctx->kv_cache[i].first.release();
                    ctx->kv_cache[i].second.release();
                }

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

                for (int i = 0; i < attn_cnt; ++i) {
                    char kname[32], vname[32];
                    std::snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                    std::snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                    ncnn::Mat k_cache, v_cache;
                    require_extract(ex.extract(kname, k_cache, 1), k_cache, kname);
                    require_extract(ex.extract(vname, v_cache, 1), v_cache, vname);
                    ctx->kv_cache[i] = { std::move(k_cache), std::move(v_cache) };
                }

                for (int i = 0; i < sconv_cnt; ++i) {
                    char name[32];
                    std::snprintf(name, sizeof(name), "out_cache_conv%d", i);
                    ncnn::Mat cache;
                    require_extract(ex.extract(name, cache), cache, name);
                    qwen_ctx->sconv_cache[i] = std::move(cache);
                }
                for (int i = 0; i < gdr_cnt; ++i) {
                    char name[32];
                    std::snprintf(name, sizeof(name), "out_cache_gdr%d", i);
                    ncnn::Mat cache;
                    require_extract(ex.extract(name, cache), cache, name);
                    qwen_ctx->gdr_cache[i] = std::move(cache);
                }

                require_extract(ex.extract("out0", decode_out), decode_out, "decoder/out0");
            }
        }

        ncnn::Mat logits_mat;
        {
            ncnn_llm::ScopedTimer t(do_perf ? &decode_perf.lm_head : nullptr);
            logits_mat = lm_head->forward(decode_out, embed_net->opt);
        }

        LlmTokenSampleConfig sample_cfg;
        sample_cfg.vocab_size = vocab_size;
        sample_cfg.temperature = cfg.temperature;
        sample_cfg.top_p = cfg.top_p;
        sample_cfg.top_k = cfg.top_k;
        sample_cfg.repetition_penalty = cfg.repetition_penalty;
        sample_cfg.do_sample = cfg.do_sample;

        int next_id = 0;
        {
            ncnn_llm::ScopedTimer t(do_perf ? &decode_perf.sampling : nullptr);
            next_id = llm_select_next_token(logits_mat, history_counts, sample_cfg, &generated_tokens);
        }

        ctx->cur_token = next_id;
        history_counts[next_id]++;
        generated_tokens.push_back(next_id);
        decode_perf.decode_tokens++;
    }

    const auto t_decode_end = std::chrono::steady_clock::now();
    decode_perf.total_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    if (perf_lvl >= 2) {
#if NCNN_BENCHMARK
        ncnn::set_layer_benchmark_active(false);
#endif
    }

    ncnn_llm::LlmPerfReport report;
    report.meta.model_path = model_path_;
    report.meta.model_type = model_type;
    report.meta.num_threads = num_threads_;
    report.meta.use_vulkan = use_vulkan_;
    report.meta.use_bf16 = use_bf16_;
    report.prefill = ctx->prefill_perf;
    report.decode = decode_perf;

#if NCNN_BENCHMARK
    if (perf_lvl >= 2) {
        report.has_layer_stats = true;
        report.layer_type_stats = ncnn::get_layer_type_benchmark_stats();
        report.layer_stats = ncnn::get_layer_benchmark_stats();
    }
#endif

    last_perf_report = report;

    if (do_perf) {
        report.print(perf_lvl >= 2);
    }

    return ctx;
}
std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::define_tools(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx, const std::vector<nlohmann::json>& tools, const std::string& system_prompt, TemplateType template_type) {

    this->tools = tools;
    std::string tool_prompt = apply_chat_template(template_type, {{"system", system_prompt}}, tools, false, false);

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
                        *ptr_r++ = (pixel[2] / 255.0f - image_mean[0]) / image_std[0];
                        *ptr_g++ = (pixel[1] / 255.0f - image_mean[1]) / image_std[1];
                        *ptr_b++ = (pixel[0] / 255.0f - image_mean[2]) / image_std[2];
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

int ncnn_llm_gpt::get_visiual_features(const ncnn::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h) const {
    if (ncnn_mat_empty(bgr)) {
        image_embeds.release();
        num_patches_w = 0;
        num_patches_h = 0;
        return 0;
    }

    if (minicpm_vision) {
        ncnn_llm::minicpm::Thumbnail thumbnail{};
        image_embeds = encode_minicpm_vision(*minicpm_vision, bgr, thumbnail);
        num_patches_w = thumbnail.patch_cols;
        num_patches_h = thumbnail.patch_rows;
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
        require_extract(ex.extract("out0", patch_embed), patch_embed, "vision_embed_patch/out0");
        memcpy(patch_embeds.row(i), patch_embed.reshape(patch_dim), patch_dim * sizeof(float));
    }

    if (vision_embed_pos) {
        ncnn::Mat pos_embeds;
        {
            ncnn::Mat grid(num_patches_w, num_patches_h);
            ncnn::Extractor ex = vision_embed_pos->create_extractor();
            ex.input("in0", grid);
            require_extract(ex.extract("out0", pos_embeds), pos_embeds, "vision_embed_pos/out0");
        }
        
        pos_embeds = reorder_patches_for_merge(pos_embeds, num_patches_h, num_patches_w, spatial_merge_size);

        ncnn::Mat emb_cos, emb_sin;
        generate_vision_rope_cache_2d(num_patches_h, num_patches_w, spatial_merge_size,
                                      10000.0f, vision_rope_section, true, emb_cos, emb_sin);

        {
            ncnn::Extractor ex = vision_encoder->create_extractor();
            ex.input("in0", patch_embeds);
            ex.input("in1", pos_embeds);
            ex.input("in2", emb_cos);
            ex.input("in3", emb_sin);
            require_extract(ex.extract("out0", image_embeds), image_embeds, "vision_encoder/out0");
            float vision_min = image_embeds[0];
            float vision_max = image_embeds[0];
            double vision_sum = 0.0;
            double vision_sq_sum = 0.0;
            for (size_t i = 0; i < image_embeds.total(); ++i) {
                const float value = image_embeds[i];
                vision_min = std::min(vision_min, value);
                vision_max = std::max(vision_max, value);
                vision_sum += value;
                vision_sq_sum += (double)value * value;
            }
            const double vision_mean = vision_sum / image_embeds.total();
            const double vision_std = std::sqrt(vision_sq_sum / image_embeds.total() - vision_mean * vision_mean);
            fprintf(stderr, "[qwen3.5 vision] shape=%d x %d range=[%g,%g] mean=%g std=%g first=%g,%g,%g,%g\n",
                    image_embeds.h, image_embeds.w, vision_min, vision_max, vision_mean, vision_std,
                    image_embeds[0], image_embeds[1], image_embeds[2], image_embeds[3]);
        }
        return 0;
    }

    ncnn::Mat emb_cos, emb_sin;
    generate_vision_rope_cache_2d(num_patches_h, num_patches_w, spatial_merge_size,
                                  10000.0f, {20, 20}, true, emb_cos, emb_sin);

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
        require_extract(ex.extract("out0", image_embeds), image_embeds, "vision_encoder/out0");
    }

    ncnn::Mat image_embeds_restored(image_embeds.w, image_embeds.h);
    for (int i = 0; i < window_index.size(); i++) {
        const int original_group_idx = window_index[i];
        const float* src_ptr = image_embeds.row(i);
        float* dst_ptr = image_embeds_restored.row(original_group_idx);
        memcpy(dst_ptr, src_ptr, image_embeds.w * sizeof(float));
    }
    image_embeds = image_embeds_restored;
    return 0;
}

void ncnn_llm_gpt::print_last_perf_report(bool include_layer_stats) const {
    last_perf_report.print(include_layer_stats);
}

std::string ncnn_llm_gpt::get_last_perf_report_str(bool include_layer_stats) const {
    return last_perf_report.to_string(include_layer_stats);
}


// MiniCPM-specific graph stages within the shared GPT vision pipeline.
namespace {
void require_minicpm(bool ok, const std::string& why) { if (!ok) throw std::runtime_error("MiniCPM vision: " + why); }
std::string minicpm_model_file(const std::string& root, const std::string& name, uintmax_t limit) {
    const std::filesystem::path relative(name);
    require_minicpm(!name.empty() && name.find('\\')==std::string::npos && name.find(':')==std::string::npos &&
            !relative.is_absolute() && !relative.has_root_path(), "manifest resource must be a relative model file");
    auto path=std::filesystem::canonical(root);
    for(const auto& part:relative) {
        require_minicpm(part!=".." && part!=".", "manifest resource traversal is unsupported");
        path/=part;
        require_minicpm(!std::filesystem::is_symlink(path), "manifest resource symlinks are unsupported");
    }
    require_minicpm(std::filesystem::is_regular_file(path), "missing model resource: "+name);
    require_minicpm(std::filesystem::file_size(path)<=limit, "model resource exceeds size limit: "+name);
    return path.string();
}
void require_minicpm_matrix(const ncnn::Mat& m, int rows, int cols, const std::string& name) {
    require_minicpm(m.dims == 2 && m.w == cols && m.h == rows && m.elempack == 1 && m.elemsize == 4u, name + " shape/precision mismatch");
}
// Row-major 2x2 windows, each flattened in [dy,dx,channel] order.
ncnn::Mat pack_minicpm_windows(const ncnn::Mat& in, int h, int w) {
    require_minicpm(h%2==0 && w%2==0, "odd merge grid");
    ncnn::Mat out(in.w*4,h*w/4);
    require_minicpm(!out.empty(), "window packing allocation failed");
    for(int y=0;y<h/2;++y) for(int x=0;x<w/2;++x) {
        float* dst=out.row(y*(w/2)+x);
        for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx)
            std::memcpy(dst+(dy*2+dx)*in.w,in.row((y*2+dy)*w+x*2+dx),in.w*sizeof(float));
    }
    return out;
}
}
std::shared_ptr<ncnn_llm::MiniCPMVisionState> ncnn_llm_gpt::load_minicpm_vision(
    const std::string& dir, const nlohmann::json& cfg, int threads) {
    auto state = std::shared_ptr<ncnn_llm::MiniCPMVisionState>(new ncnn_llm::MiniCPMVisionState);
    auto& mode = state->mode;
    auto& hidden = state->hidden;
    auto& output_hidden = state->output_hidden;
    auto& patch = state->patch;
    auto& position_side = state->position_side;
    auto& scale = state->scale;
    auto& slice_mode = state->slice_mode;
    auto& max_slices = state->max_slices;
    auto& markers = state->markers;
    auto& graphs = state->graphs;
    auto& position = state->position;
    std::ifstream file(minicpm_model_file(dir,cfg.at("manifest").get<std::string>(),1024*1024));
    require_minicpm(file.good(), "missing manifest");
    nlohmann::json manifest; file>>manifest;
    validate_minicpm_vision_manifest(manifest);
    mode=manifest.at("downsample_mode").get<std::string>();
    require_minicpm(mode=="16x", "only 16x component graphs are supported");
    hidden=manifest.at("hidden_size"); output_hidden=manifest.at("output_hidden_size");
    patch=manifest.at("patch_size"); position_side=manifest.at("position_grid_size");
    std::fprintf(stderr, "[MiniCPM vision] Preprocessing: official MiniCPMV4_6ImageProcessorPil-compatible, Pillow 12.3 bicubic; default Torchvision resize is a different backend.\n");
    scale=cfg.value("scale_resolution",448); slice_mode=cfg.value("slice_mode",true); max_slices=cfg.value("max_slice_nums",9);
    require_minicpm(scale>0 && scale<=896 && max_slices>=1 && max_slices<=9, "unsupported thumbnail scale or slice limit");
    auto ids=cfg.at("marker_ids");
    markers={ids.at("image"),ids.at("image_start"),ids.at("image_end"),ids.at("slice_start"),ids.at("slice_end"),ids.at("newline")};
    std::set<int> unique_ids;
    for(int id:{markers.image,markers.image_start,markers.image_end,markers.slice_start,markers.slice_end,markers.newline}) {
        require_minicpm(id>=0, "negative marker token id"); unique_ids.insert(id);
    }
    require_minicpm(unique_ids.size()==6, "marker token ids must be distinct");
    for(const char* name:{"patch_embed","encoder_pre","window_attn","window_mlp","encoder_post","projector"}) {
        auto net=std::make_shared<ncnn::Net>();
        net->opt.use_vulkan_compute=false; net->opt.use_packing_layout=false;
        net->opt.use_fp16_packed=false; net->opt.use_fp16_storage=false; net->opt.use_fp16_arithmetic=false; net->opt.use_bf16_storage=false;
        if(threads>0) net->opt.num_threads=threads;
        const auto& component=manifest.at("components").at(name);
        load_net(*net, std::string("MiniCPM vision/")+name,
            minicpm_model_file(dir,component.at("param").get<std::string>(),16*1024*1024),
            minicpm_model_file(dir,component.at("bin").get<std::string>(),uintmax_t(8)*1024*1024*1024));
        graphs[name]=net;
    }
    auto pos=manifest.at("position_embeddings");
    require_minicpm(pos.at("rows").get<int>()==position_side*position_side && pos.at("cols").get<int>()==hidden,"position table shape mismatch");
    position.resize(static_cast<size_t>(position_side)*position_side*hidden);
    std::ifstream weights(minicpm_model_file(dir,pos.at("file").get<std::string>(),position.size()*sizeof(float)),std::ios::binary);
    require_minicpm(weights.good() && bool(weights.read(reinterpret_cast<char*>(position.data()),position.size()*sizeof(float))),"truncated position table");
    require_minicpm(weights.peek()==std::char_traits<char>::eof(),"oversized position table");
    return state;
}
static ncnn::Mat run_minicpm_graph(const std::map<std::string, std::shared_ptr<ncnn::Net>>& graphs,
                                    const std::string& name, const ncnn::Mat& input) {
    auto ex=graphs.at(name)->create_extractor(); ncnn::Mat out;
    require_minicpm(ex.input("in0",input)==0,"input failed: "+name);
    require_minicpm(ex.extract("out0",out)==0 && !out.empty(),"extraction failed: "+name);
    return out;
}
void ncnn_llm_gpt::validate_minicpm_vision_manifest(const nlohmann::json& m) {
    require_minicpm(m.at("type")=="minicpm_v47" && m.at("format_version")==1 && m.at("weight_dtype")=="float32" &&
            m.at("scope")=="single_image_thumbnail" && m.at("downsample_mode")=="16x", "unsupported manifest format/precision/scope");
    const int h=m.at("hidden_size"), out=m.at("output_hidden_size"), side=m.at("position_grid_size");
    require_minicpm(h>0 && h<=2048 && out>0 && out<=8192 && side>0 && side<=128 && m.at("patch_size")==14,
            "unsupported or excessive architecture dimensions");
    require_minicpm(m.at("window_kernel_size")==nlohmann::json::array({2,2}) &&
            m.at("merge_kernel_size")==nlohmann::json::array({2,2}), "only 2x2 window and projector merges are supported");
    const int layers=m.at("num_hidden_layers"), insertion=m.at("insert_layer_id");
    require_minicpm(layers>1 && layers<=64 && insertion>=0 && insertion<layers-1, "invalid encoder partition");
    require_minicpm(m.at("position_embeddings").at("rows").get<int>()==side*side &&
            m.at("position_embeddings").at("cols").get<int>()==h, "position table shape mismatch");
}
void ncnn_llm_gpt::validate_minicpm_thumbnail(const ncnn_llm::minicpm::Thumbnail& t) {
    require_minicpm(t.patch_rows>0 && t.patch_cols>0 && static_cast<int64_t>(t.patch_rows)*t.patch_cols <= 4096,
            "experimental thumbnail exceeds 4096 ViT patches or has invalid dimensions; extreme aspect ratios are unsupported");
}
ncnn::Mat ncnn_llm_gpt::encode_minicpm_vision(const ncnn_llm::MiniCPMVisionState& state,
    const ncnn::Mat& bgr, ncnn_llm::minicpm::Thumbnail& t) {
    const auto& mode = state.mode;
    const auto& hidden = state.hidden;
    const auto& output_hidden = state.output_hidden;
    const auto& patch = state.patch;
    const auto& position_side = state.position_side;
    const auto& scale = state.scale;
    const auto& slice_mode = state.slice_mode;
    const auto& max_slices = state.max_slices;
    const auto& position = state.position;
    require_minicpm(!bgr.empty() && bgr.dims==3 && bgr.c==3 && bgr.elemsize==1u && bgr.elempack==1,"expected interleaved BGR uint8 image");
    require_minicpm(bgr.w>0 && bgr.h>0 && bgr.w<=16384 && bgr.h<=16384 && static_cast<int64_t>(bgr.w)*bgr.h<=16000000, "input image exceeds the experimental dimension limit");
    t=ncnn_llm::minicpm::plan_thumbnail(bgr.h,bgr.w,patch,scale,mode,slice_mode,max_slices);
    validate_minicpm_thumbnail(t);
    const unsigned char* raw=static_cast<const unsigned char*>(bgr.data);
    std::vector<uint8_t> source(raw,raw+static_cast<size_t>(bgr.w)*bgr.h*3);
    const auto rgb=ncnn_llm::minicpm::resize_bicubic_rgb(source,bgr.h,bgr.w,t.height,t.width,true);
    ncnn::Mat pixels(t.width,t.height,3);
    require_minicpm(!pixels.empty(), "pixel allocation failed");
    for(int c=0;c<3;++c) {
        float* values=pixels.channel(c);
        for(size_t i=0;i<static_cast<size_t>(t.width)*t.height;++i)
            values[i]=(static_cast<float>(rgb[i*3+c])/255.f-0.5f)/0.5f;
    }
    ncnn::Mat x=run_minicpm_graph(state.graphs, "patch_embed",pixels);
    require_minicpm_matrix(x,t.patch_rows*t.patch_cols,hidden,"patch_embed");
    auto positions=ncnn_llm::minicpm::nearest_position_ids(t.patch_rows,t.patch_cols,position_side);
    for(int i=0;i<x.h;++i) { float* row=x.row(i); const float* p=position.data()+positions[i]*hidden; for(int c=0;c<hidden;++c) row[c]+=p[c]; }
    x=run_minicpm_graph(state.graphs, "encoder_pre",x); require_minicpm_matrix(x,t.patch_rows*t.patch_cols,hidden,"encoder_pre");
    ncnn::Mat windows=pack_minicpm_windows(x,t.patch_rows,t.patch_cols);
    ncnn::Mat attended(hidden*4,windows.h);
    require_minicpm(!attended.empty(), "window attention allocation failed");
    for(int i=0;i<windows.h;++i) {
        // A row view has no refcount. ncnn's light-mode in-place dispatcher
        // needs an owned input before the attention branch's LayerNorm.
        ncnn::Mat one=ncnn::Mat(hidden,4,static_cast<void*>(windows.row(i))).clone();
        require_minicpm(!one.empty(), "window input allocation failed");
        ncnn::Mat result=run_minicpm_graph(state.graphs, "window_attn",one); require_minicpm_matrix(result,4,hidden,"window_attn");
        std::memcpy(attended.row(i),result.data,4*hidden*sizeof(float));
    }
    x=run_minicpm_graph(state.graphs, "window_mlp",attended); require_minicpm_matrix(x,windows.h,hidden,"window_mlp");
    for(int i=0;i<x.h;++i) { float* row=x.row(i); const float* a=attended.row(i); for(int c=0;c<hidden;++c) row[c]+=(a[c]+a[hidden+c]+a[2*hidden+c]+a[3*hidden+c])*0.25f; }
    x=run_minicpm_graph(state.graphs, "encoder_post",x); require_minicpm_matrix(x,windows.h,hidden,"encoder_post");
    x=run_minicpm_graph(state.graphs, "projector",pack_minicpm_windows(x,t.patch_rows/2,t.patch_cols/2));
    require_minicpm_matrix(x,t.token_count(),output_hidden,"projector");
    return x;
}
