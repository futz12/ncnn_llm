#pragma once

// Versioned build capability for downstream adapters; absent in older runtimes.
#define NCNN_LLM_MINICPM_V47_RUNTIME 1

#include <array>
#include <cassert>
#include <cstdio>
#include <exception>
#include <functional>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"
#include "utils/prompt.h"
#include "kernel/gdr.h"
#include "utils/image_utils.h"
#include "utils/perf_profiler.h"
#include "kernel/lm_head.h"
#include <map>

using nlohmann::json;

struct GenerateConfig {
    int max_new_tokens = 4096;
    float temperature = 0.3f;
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    int do_sample = 1;
    bool enable_thinking = false;

    std::function<nlohmann::json(const nlohmann::json&)> tool_callback = nullptr;

    bool debug = false;
    bool enable_perf = false;
    int perf_level = 0;
};

inline ncnn::Mat clone_kvcache_mat(const ncnn::Mat& src, ncnn::Allocator* allocator) {
    if (src.empty()) return ncnn::Mat();
    if (src.dims == 3 && src.w > 0) {
        const int capacity = (int)(src.cstep / src.w);
        ncnn::Mat dst(src.w, capacity, src.c, src.elemsize, src.elempack, allocator);
        dst.h = src.h;
        if (src.data && dst.data) {
            memcpy(dst.data, src.data, src.c * src.cstep * src.elemsize);
        }
        return dst;
    }
    return src.clone(allocator);
}

class ncnn_llm_gpt_ctx {
public:
    virtual ~ncnn_llm_gpt_ctx() {
        kv_cache.clear();
    }
    
    virtual std::shared_ptr<ncnn_llm_gpt_ctx> clone() const = 0;
    
    std::shared_ptr<ncnn::UnlockedPoolAllocator> kvcache_allocator;
    KVCache kv_cache;
    int cur_token = 0;
    int position_id = 0;
    int image_count = 0;
    ncnn_llm::PrefillPerfStats prefill_perf;
};

class ncnn_llm_gpt_base_ctx : public ncnn_llm_gpt_ctx {
public:
    std::shared_ptr<ncnn_llm_gpt_ctx> clone() const override {
        auto dst = std::make_shared<ncnn_llm_gpt_base_ctx>();
        dst->kvcache_allocator = std::make_shared<ncnn::UnlockedPoolAllocator>();
        dst->kvcache_allocator->set_size_compare_ratio(0.f);
        dst->kv_cache.resize(kv_cache.size());
        for (size_t i = 0; i < kv_cache.size(); ++i) {
            dst->kv_cache[i].first = clone_kvcache_mat(kv_cache[i].first, dst->kvcache_allocator.get());
            dst->kv_cache[i].second = clone_kvcache_mat(kv_cache[i].second, dst->kvcache_allocator.get());
        }
        dst->cur_token = cur_token;
        dst->position_id = position_id;
        dst->image_count = image_count;
        dst->prefill_perf = prefill_perf;
        return dst;
    }
};

class qwen3_5_ctx : virtual public ncnn_llm_gpt_ctx {
public:
    std::vector<ncnn::Mat> sconv_cache;
    std::vector<ncnn::Mat> gdr_cache;
    
    ~qwen3_5_ctx() override {
        sconv_cache.clear();
        gdr_cache.clear();
        kv_cache.clear();
    }
    
    std::shared_ptr<ncnn_llm_gpt_ctx> clone() const override {
        auto dst = std::make_shared<qwen3_5_ctx>();
        dst->kvcache_allocator = std::make_shared<ncnn::UnlockedPoolAllocator>();
        dst->kvcache_allocator->set_size_compare_ratio(0.f);
        dst->kv_cache.resize(kv_cache.size());
        for (size_t i = 0; i < kv_cache.size(); ++i) {
            dst->kv_cache[i].first = clone_kvcache_mat(kv_cache[i].first, dst->kvcache_allocator.get());
            dst->kv_cache[i].second = clone_kvcache_mat(kv_cache[i].second, dst->kvcache_allocator.get());
        }
        dst->sconv_cache.resize(sconv_cache.size());
        for (size_t i = 0; i < sconv_cache.size(); ++i) {
            dst->sconv_cache[i] = sconv_cache[i].clone();
        }
        dst->gdr_cache.resize(gdr_cache.size());
        for (size_t i = 0; i < gdr_cache.size(); ++i) {
            dst->gdr_cache[i] = gdr_cache[i].clone();
        }
        dst->cur_token = cur_token;
        dst->position_id = position_id;
        dst->image_count = image_count;
        dst->prefill_perf = prefill_perf;
        return dst;
    }
};

class ncnn_llm_gpt;

namespace ncnn_llm {
// Extra graph partitions and learned positions required by MiniCPM's windowed ViT.
// Execution is part of ncnn_llm_gpt's existing vision feature pipeline.
class MiniCPMVisionState {
    friend class ::ncnn_llm_gpt;
    MiniCPMVisionState() = default;
    MiniCPMVisionState(const MiniCPMVisionState&) = delete;
    MiniCPMVisionState& operator=(const MiniCPMVisionState&) = delete;
    minicpm::MarkerIds markers;
    std::string mode;
    std::map<std::string, std::shared_ptr<ncnn::Net>> graphs;
    std::vector<float> position;
    int hidden = 0, output_hidden = 0, patch = 0, position_side = 0, scale = 0, max_slices = 0;
    bool slice_mode = false;
};
}

class ncnn_llm_gpt {
private:
    std::shared_ptr<ncnn::Net> decoder_net;
    std::shared_ptr<ncnn::Net> embed_net;
    std::shared_ptr<ncnn::Net> proj_out_net;
    std::shared_ptr<ncnn_llm::LlmHead> lm_head;
    std::shared_ptr<ncnn::Net> vision_embed_patch;
    std::shared_ptr<ncnn::Net> vision_embed_pos;
    std::shared_ptr<ncnn::Net> vision_encoder;
    std::shared_ptr<BpeTokenizer> bpe;
    std::shared_ptr<ncnn_llm::MiniCPMVisionState> minicpm_vision;

protected:
    std::string model_type;
    int bos = 0;
    int eos = 0;
    std::unordered_set<int> eos_ids;
    int tool_call_id = -1;
    int tool_call_end_id = -1;
    int think_id = -1;
    int think_end_id = -1;
    int attn_cnt = 32;
    int sconv_cnt = 0;
    int gdr_cnt = 0;
    int rope_head_dim = 64;

    enum RoPE_Type {
        RoPE = 0,
        LongRoPE = 1,
        NTK_RoPE = 2,
        YARN_RoPE = 3
    } rope_type;
    float rope_theta = 100000.0f;

    RopeScalingParams ntk_scaling_params;

    std::vector<float> short_factor;
    std::vector<float> long_factor;
    int original_max_position_embeddings = 0;

    int image_pad_id = -1;
    int patch_size = 14;
    int patch_dim = 1280;
    int max_num_patches = 49152;
    int spatial_merge_size = 2;

    enum Vision_Type {
        VISION_CLOSE = 0,
        VISION_VIT = 1,
        VISION_QWEN3_5_VL = 2,
        VISION_MINICPM_V4_7 = 3
    } vision_type;

    enum VisionRoPE_Type {
        mRoPE = 0
    } vision_rope_type;

    std::vector<int> mrope_section;
    std::vector<int> vision_rope_section = {16, 16};
    std::vector<nlohmann::json> tools;

public:
    // Vision-only entry points also permit parity checks without loading the LLM.
    static std::shared_ptr<ncnn_llm::MiniCPMVisionState> load_minicpm_vision(
        const std::string& directory, const nlohmann::json& config, int threads);
    static void validate_minicpm_vision_manifest(const nlohmann::json& manifest);
    static void validate_minicpm_thumbnail(const ncnn_llm::minicpm::Thumbnail& thumbnail);
    static ncnn::Mat encode_minicpm_vision(const ncnn_llm::MiniCPMVisionState& state,
        const ncnn::Mat& bgr, ncnn_llm::minicpm::Thumbnail& thumbnail);

    ncnn_llm_gpt(const std::string& model_path, bool use_vulkan = false, int num_threads = 0, int vulkan_device = 0, bool use_bf16 = true, bool force_naive = false);

    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text, const ncnn::Mat& bgr, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> generate(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in, const GenerateConfig& cfg, std::function<void(const std::string&)> callback) const;

    std::shared_ptr<ncnn_llm_gpt_ctx> define_tools(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx, const std::vector<nlohmann::json>& tools, const std::string& system_prompt = "You are a helpful assistant.", TemplateType template_type = TemplateType::CHATML);

    std::string image_prompt_placeholder() const { return minicpm_vision ? "<image>\n" : "<|vision_start|><|image_pad|><|vision_end|>"; }
    bool supports_tool_calling() const { return tool_call_id >= 0; }
    bool supports_vision() const { return vision_type != Vision_Type::VISION_CLOSE; }
    const std::string& get_model_type() const { return model_type; }

    std::string model_path_;
    bool use_vulkan_ = false;
    int num_threads_ = 0;
    int vulkan_device_ = 0;
    bool use_bf16_ = true;
    mutable ncnn_llm::LlmPerfReport last_perf_report;

    const ncnn_llm::LlmPerfReport& get_last_perf_report() const { return last_perf_report; }
    void print_last_perf_report(bool include_layer_stats = true) const;
    std::string get_last_perf_report_str(bool include_layer_stats = true) const;

    template<typename T>
    static constexpr const char* json_type_name() {
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long>) return "integer";
        else if constexpr (std::is_same_v<T, bool>) return "boolean";
        else if constexpr (std::is_floating_point_v<T>) return "number";
        else return "string";
    }

    template<typename Ret, typename... Args>
    static nlohmann::json make_function_tool(const std::string& name, const std::string& description, const std::array<std::string, sizeof...(Args)>& arg_names)
    {
        assert(arg_names.size() == sizeof...(Args));
        nlohmann::json properties = nlohmann::json::object();
        size_t idx = 0;
        ((
            properties[arg_names[idx]] = nlohmann::json{{"type", json_type_name<Args>()}, {"description", ""}},
            ++idx
        ), ...);

        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", description},
                {"parameters", {
                    {"type", "object"},
                    {"properties", properties},
                    {"required", arg_names}
                }}
            }}
        };
    }

private:
    int get_scaled_image_size(float scale, int size, int effective_patch_size) const;
    void get_image_size_for_patches(int image_height, int image_width, int patch_size, int max_num_patches, int& target_height, int& target_width) const;
    ncnn::Mat bgr_to_pixel_values(const ncnn::Mat& bgr) const;
    ncnn::Mat reorder_patches_for_merge(const ncnn::Mat& pixel_values, int h_patches, int w_patches, int merge_size = 2) const;
    void get_window_index(int num_patches_w, int num_patches_h, std::vector<int>& window_index, std::vector<int>& cu_window_seqlens) const;
    int get_visiual_features(const ncnn::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h) const;
};
