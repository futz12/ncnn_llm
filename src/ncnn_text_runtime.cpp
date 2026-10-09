#include "kernel/lm_head.h"
#include "ncnn_text_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <utility>

#include "sampling.h"

ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, const std::vector<int>& input_ids) {
    ncnn::Mat input_ids_mat((int)input_ids.size(), 1, (void*)input_ids.data());
    input_ids_mat = input_ids_mat.clone();

    ncnn::Mat token_embed;
    ncnn::Extractor ex = embed_net.create_extractor();
    ex.input("in0", input_ids_mat);
    ex.extract("out0", token_embed);
    return token_embed;
}

ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, int token_id) {
    // Do not view a 4-byte stack int as a [1,1] Mat and then clone it:
    // Mat::clone() copies total() * elemsize bytes, and Mat::total() returns
    // cstep * c where cstep is the 16-byte aligned step. For a [1,1] float Mat
    // cstep == alignSize(4, 16) / 4 == 4, so the copy reads 16 bytes from a
    // 4-byte object (AddressSanitizer: stack-buffer-overflow).
    // Back it with a 16-byte aligned buffer of the same shape instead.
    alignas(16) int token_id_buf[4] = {0, 0, 0, 0};
    token_id_buf[0] = token_id;
    ncnn::Mat input_id_mat(1, 1, (void*)token_id_buf);
    input_id_mat = input_id_mat.clone();

    ncnn::Mat token_embed;
    ncnn::Extractor ex = embed_net.create_extractor();
    ex.input("in0", input_id_mat);
    ex.extract("out0", token_embed);
    return token_embed;
}

ncnn::Mat llm_run_decoder_with_kv(ncnn::Net& decoder_net,
                                  const ncnn::Mat& embeds,
                                  const ncnn::Mat& mask,
                                  const ncnn::Mat& cos_cache,
                                  const ncnn::Mat& sin_cache,
                                  KVCache& kv_cache,
                                  int attn_cnt,
                                  bool is_prefill,
                                  ncnn::Allocator* kvcache_allocator,
                                  int max_seqlen_hint) {
    ncnn::Mat decode_out;
    ncnn::Extractor ex = decoder_net.create_extractor();
    if (kvcache_allocator) {
        ex.set_kvcache_allocator(kvcache_allocator);
    }
    if (max_seqlen_hint > 0) {
        ex.set_kvcache_max_seqlen_hint(max_seqlen_hint);
    }
    ex.input("in0", embeds);
    ex.input("in1", mask);
    ex.input("in2", cos_cache);
    ex.input("in3", sin_cache);

    if (!is_prefill) {
        for (int i = 0; i < attn_cnt; i++) {
            char name_k_in[16], name_v_in[16];
            std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
            std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
            ex.input(name_k_in, kv_cache[i].first);
            ex.input(name_v_in, kv_cache[i].second);
            kv_cache[i].first.release();
            kv_cache[i].second.release();
        }
    }

    for (int i = 0; i < attn_cnt; i++) {
        char name_k_out[32], name_v_out[32];
        std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
        std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
        ncnn::Mat k_cache, v_cache;
        ex.extract(name_k_out, k_cache, 1);
        ex.extract(name_v_out, v_cache, 1);
        if (is_prefill) {
            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        } else {
            kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }
    }

    ex.extract("out0", decode_out);
    return decode_out;
}

ncnn::Mat llm_run_lm_head(const ncnn_llm::LlmHead& lm_head, const ncnn::Mat& hidden_states, const ncnn::Option& opt) {
    return lm_head.forward(hidden_states, opt);
}
ncnn::Mat llm_run_lm_head(ncnn::Net& lm_head_net, const ncnn::Mat& hidden_states) {
    ncnn::Mat logits;
    ncnn::Extractor ex = lm_head_net.create_extractor();
    ex.input("in0", hidden_states);
    ex.extract("out0", logits);
    return logits;
}

int llm_select_next_token(const ncnn::Mat& logits,
                          const std::unordered_map<int, int>& history_counts,
                          const LlmTokenSampleConfig& cfg,
                          const std::vector<int>* generated_tokens) {
    if (logits.empty() || logits.w <= 0 || !logits.data) {
        return 0;
    }
    const int vocab_size = cfg.vocab_size > 0
        ? std::min(cfg.vocab_size, logits.w)
        : logits.w;
    if (vocab_size <= 0) {
        return 0;
    }
    thread_local std::vector<float> scores;
    scores.resize(vocab_size);
    std::memcpy(scores.data(), logits.data, sizeof(float) * vocab_size);

    for (const auto& kv : history_counts) {
        int t = kv.first;
        int count = kv.second;
        if (t < 0 || t >= vocab_size || count <= 0) continue;
        float penalty = (cfg.repetition_penalty > 1.0f)
            ? std::pow(cfg.repetition_penalty, (float)std::min(count, 16))
            : 1.0f;
        if (scores[t] < 0) {
            scores[t] *= penalty;
        } else {
            scores[t] /= penalty;
        }
    }

    if (generated_tokens && !generated_tokens->empty()) {
        const auto& gt = *generated_tokens;
        const int n = (int)gt.size();
        for (int L = 1; L <= 32 && n >= 2 * L; ++L) {
            bool match = true;
            for (int i = 0; i < L; ++i) {
                if (gt[n - 1 - i] != gt[n - 1 - L - i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                int next_cycle_token = gt[n - L];
                if (next_cycle_token >= 0 && next_cycle_token < vocab_size) {
                    if (scores[next_cycle_token] < 0) {
                        scores[next_cycle_token] *= 2.0f;
                    } else {
                        scores[next_cycle_token] /= 2.0f;
                    }
                }
            }
        }
    }

    if (cfg.do_sample != 1 || cfg.temperature <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    softmax_vec(scores, cfg.temperature);
    if (cfg.top_k > 0) apply_top_k(scores, cfg.top_k);
    if (cfg.top_p < 1.0f) apply_top_p(scores, cfg.top_p);

    const float sum = std::accumulate(scores.begin(), scores.end(), 0.0f);
    if (!std::isfinite(sum) || sum <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    return sample_from_probs(scores);
}

int llm_select_next_token(const ncnn::Mat& logits,
                          const std::unordered_set<int>& history,
                          const LlmTokenSampleConfig& cfg) {
    std::unordered_map<int, int> counts;
    for (int t : history) counts[t] = 1;
    return llm_select_next_token(logits, counts, cfg, nullptr);
}


