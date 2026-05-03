#include "ncnn_embedding.h"
#include "utils/image_utils.h"
#include <cmath>
#include <algorithm>

ncnn_embedding::ncnn_embedding(const std::string& model_path, bool use_vulkan, int num_threads, int vulkan_device) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            ifs >> config;
        }

        if (config.contains("model_type")) {
            model_type_ = config["model_type"].get<std::string>();
        }

        text_encoder_net = std::make_shared<ncnn::Net>();

        if (model_type_ == "clip") {
            vision_encoder_net = std::make_shared<ncnn::Net>();
        }

        if (num_threads > 0) {
            text_encoder_net->opt.num_threads = num_threads;
            if (vision_encoder_net) {
                vision_encoder_net->opt.num_threads = num_threads;
            }
            num_threads_ = num_threads;
        }

        if (use_vulkan) {
            printf("[ncnn_embedding] Vulkan enabled, using device %d\n", vulkan_device >= 0 ? vulkan_device : 0);
#if NCNN_VULKAN
            if (vulkan_device >= 0) {
                text_encoder_net->opt.vulkan_device_index = vulkan_device;
                if (vision_encoder_net) {
                    vision_encoder_net->opt.vulkan_device_index = vulkan_device;
                }
            }
#endif
            text_encoder_net->opt.use_bf16_storage = true;
            text_encoder_net->opt.use_fp16_arithmetic = false;
            text_encoder_net->opt.use_fp16_storage = false;
            text_encoder_net->opt.use_fp16_packed = false;
            text_encoder_net->opt.use_vulkan_compute = true;
            if (vision_encoder_net) {
                vision_encoder_net->opt.use_bf16_storage = true;
                vision_encoder_net->opt.use_fp16_arithmetic = false;
                vision_encoder_net->opt.use_fp16_storage = false;
                vision_encoder_net->opt.use_fp16_packed = false;
                vision_encoder_net->opt.use_vulkan_compute = true;
            }
            use_vulkan_ = true;
        } else {
            printf("[ncnn_embedding] Vulkan disabled, using CPU only\n");
        }

        std::string encoder_param, encoder_bin;
        if (model_type_ == "clip") {
            encoder_param = model_path + "/" + config["params"]["text_encoder_param"].get<std::string>();
            encoder_bin = model_path + "/" + config["params"]["text_encoder_bin"].get<std::string>();
        } else {
            encoder_param = model_path + "/" + config["params"]["encoder_param"].get<std::string>();
            encoder_bin = model_path + "/" + config["params"]["encoder_bin"].get<std::string>();
        }

        printf("Loading embedding model from %s\n", model_path.c_str());
        printf("  model_type: %s\n", model_type_.c_str());
        printf("  text encoder param: %s\n", encoder_param.c_str());
        printf("  text encoder bin: %s\n", encoder_bin.c_str());

        text_encoder_net->load_param(encoder_param.c_str());
        text_encoder_net->load_model(encoder_bin.c_str());

        if (model_type_ == "clip") {
            std::string vision_param = model_path + "/" + config["params"]["vision_encoder_param"].get<std::string>();
            std::string vision_bin = model_path + "/" + config["params"]["vision_encoder_bin"].get<std::string>();
            printf("  vision encoder param: %s\n", vision_param.c_str());
            printf("  vision encoder bin: %s\n", vision_bin.c_str());
            vision_encoder_net->load_param(vision_param.c_str());
            vision_encoder_net->load_model(vision_bin.c_str());
        }

        std::string tokenizer_type = "bpe";
        if (config["tokenizer"].contains("type")) {
            tokenizer_type = config["tokenizer"]["type"].get<std::string>();
        }
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();

        if (tokenizer_type == "unigram") {
            SpecialTokensConfig spec;
            if (config["tokenizer"].contains("bos")) {
                spec.bos_token = config["tokenizer"]["bos"].get<std::string>();
            }
            if (config["tokenizer"].contains("eos")) {
                spec.eos_token = config["tokenizer"]["eos"].get<std::string>();
            }
            if (config["tokenizer"].contains("pad")) {
                spec.pad_token = config["tokenizer"]["pad"].get<std::string>();
            }
            if (config["tokenizer"].contains("unk")) {
                spec.unk_token = config["tokenizer"]["unk"].get<std::string>();
            }
            if (config["tokenizer"].contains("cls")) {
                spec.cls_token = config["tokenizer"]["cls"].get<std::string>();
            }
            if (config["tokenizer"].contains("sep")) {
                spec.sep_token = config["tokenizer"]["sep"].get<std::string>();
            }
            if (config["tokenizer"].contains("mask")) {
                spec.mask_token = config["tokenizer"]["mask"].get<std::string>();
            }
            unigram_tokenizer = std::make_shared<UnigramTokenizer>(UnigramTokenizer::LoadFromFile(
                vocab_file, spec, true, true, -10.0
            ));
        } else {
            std::string merges_file;
            if (config["tokenizer"].contains("merges_file")) {
                merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();
            }
            bpe_tokenizer = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
                vocab_file, merges_file, SpecialTokensConfig{}, false, true, tokenizer_type == "bbpe"
            ));
        }

        if (config["setting"].contains("embed_dim")) {
            text_embed_dim_ = config["setting"]["embed_dim"].get<int>();
        }
        if (config["setting"].contains("text_embed_dim")) {
            text_embed_dim_ = config["setting"]["text_embed_dim"].get<int>();
        }
        if (config["setting"].contains("vision_embed_dim")) {
            vision_embed_dim_ = config["setting"]["vision_embed_dim"].get<int>();
        }
        if (config["setting"].contains("image_size")) {
            image_size_ = config["setting"]["image_size"].get<int>();
        }
        if (config["setting"].contains("max_seq_len")) {
            max_seq_len_ = config["setting"]["max_seq_len"].get<int>();
        }
        if (config["setting"].contains("image_mean")) {
            auto& mean_arr = config["setting"]["image_mean"];
            image_mean_.clear();
            for (auto& v : mean_arr) {
                image_mean_.push_back(v.get<float>());
            }
        }
        if (config["setting"].contains("image_std")) {
            auto& std_arr = config["setting"]["image_std"];
            image_std_.clear();
            for (auto& v : std_arr) {
                image_std_.push_back(v.get<float>());
            }
        }

        if (config["setting"].contains("rope")) {
            auto& rope_config = config["setting"]["rope"];
            if (rope_config.contains("type")) {
                rope_type_ = rope_config["type"].get<std::string>();
            }
            if (rope_config.contains("rope_head_dim")) {
                rope_head_dim_ = rope_config["rope_head_dim"].get<int>();
            }
            if (rope_config.contains("rope_theta")) {
                rope_theta_ = rope_config["rope_theta"].get<float>();
            }
        } else {
            if (config["setting"].contains("rope_head_dim")) {
                rope_head_dim_ = config["setting"]["rope_head_dim"].get<int>();
            }
            if (config["setting"].contains("rope_theta")) {
                rope_theta_ = config["setting"]["rope_theta"].get<float>();
            }
        }

        printf("  text_embed_dim: %d\n", text_embed_dim_);
        printf("  vision_embed_dim: %d\n", vision_embed_dim_);
        if (model_type_ == "clip") {
            printf("  image_size: %d\n", image_size_);
        }
        printf("  rope_type: %s\n", rope_type_.c_str());
        printf("  rope_head_dim: %d\n", rope_head_dim_);
        printf("  rope_theta: %f\n", rope_theta_);

    } catch (const std::exception& e) {
        fprintf(stderr, "Error loading embedding model: %s\n", e.what());
        ok_ = false;
    }
}

ncnn::Mat ncnn_embedding::mean_pool(const ncnn::Mat& embeddings) const {
    int seq_len = embeddings.h;
    int dim = embeddings.w;

    ncnn::Mat pooled(dim);
    pooled.fill(0.0f);

    for (int i = 0; i < seq_len; ++i) {
        const float* row = embeddings.row(i);
        float* pooled_ptr = pooled;
        for (int j = 0; j < dim; ++j) {
            pooled_ptr[j] += row[j];
        }
    }

    float scale = 1.0f / static_cast<float>(seq_len);
    float* pooled_ptr = pooled;
    for (int j = 0; j < dim; ++j) {
        pooled_ptr[j] *= scale;
    }

    return pooled;
}

ncnn::Mat ncnn_embedding::normalize(const ncnn::Mat& vec) const {
    int dim = vec.w;
    const float* ptr = vec;

    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        norm += ptr[i] * ptr[i];
    }
    norm = std::sqrt(norm);

    ncnn::Mat normalized(dim);
    float* out_ptr = normalized;
    if (norm > 1e-10f) {
        for (int i = 0; i < dim; ++i) {
            out_ptr[i] = ptr[i] / norm;
        }
    } else {
        for (int i = 0; i < dim; ++i) {
            out_ptr[i] = ptr[i];
        }
    }

    return normalized;
}

ncnn::Mat ncnn_embedding::preprocess_image(const std::vector<unsigned char>& image_data, int width, int height, int channels) const {
    ncnn::Mat output(image_size_, image_size_, 3);

    float scale_x = static_cast<float>(width) / image_size_;
    float scale_y = static_cast<float>(height) / image_size_;

    for (int y = 0; y < image_size_; ++y) {
        for (int x = 0; x < image_size_; ++x) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, width - 1);
            int y1 = std::min(y0 + 1, height - 1);

            float dx = src_x - x0;
            float dy = src_y - y0;

            for (int c = 0; c < 3; ++c) {
                int src_c = std::min(c, channels - 1);
                int idx00 = (y0 * width + x0) * channels + src_c;
                int idx01 = (y0 * width + x1) * channels + src_c;
                int idx10 = (y1 * width + x0) * channels + src_c;
                int idx11 = (y1 * width + x1) * channels + src_c;

                float v00 = static_cast<float>(image_data[idx00]);
                float v01 = static_cast<float>(image_data[idx01]);
                float v10 = static_cast<float>(image_data[idx10]);
                float v11 = static_cast<float>(image_data[idx11]);

                float v = v00 * (1 - dx) * (1 - dy) +
                          v01 * dx * (1 - dy) +
                          v10 * (1 - dx) * dy +
                          v11 * dx * dy;

                v = (v / 255.0f - image_mean_[c]) / image_std_[c];

                output.channel(c).row(y)[x] = v;
            }
        }
    }

    return output;
}

std::vector<float> ncnn_embedding::encode_text(const std::string& text) const {
    if (!ok_) {
        return {};
    }

    std::vector<int> token_ids;
    if (unigram_tokenizer) {
        token_ids = unigram_tokenizer->encode(text, true, true);
    } else if (bpe_tokenizer) {
        token_ids = bpe_tokenizer->encode(text, false, false);
    }

    int seq_len = static_cast<int>(token_ids.size());
    if (seq_len == 0) {
        return std::vector<float>(text_embed_dim_, 0.0f);
    }

    if (seq_len > max_seq_len_) {
        token_ids.resize(max_seq_len_);
        seq_len = max_seq_len_;
    }

    ncnn::Mat input_ids(seq_len, 1, 1);
    for (int i = 0; i < seq_len; ++i) {
        ((int*)input_ids.data)[i] = token_ids[i];
    }

    ncnn::Mat cos_cache, sin_cache;
    if (rope_type_ == "RoPE_full") {
        generate_rope_embed_cache_full(seq_len, rope_head_dim_, 0, cos_cache, sin_cache, rope_theta_);
    } else {
        generate_rope_embed_cache(seq_len, rope_head_dim_, 0, cos_cache, sin_cache, rope_theta_);
    }

    ncnn::Mat embeddings;
    {
        ncnn::Extractor ex = text_encoder_net->create_extractor();
        ex.input("in0", input_ids);
        ex.input("in1", cos_cache);
        ex.input("in2", sin_cache);
        ex.extract("out0", embeddings);
    }

    if (embeddings.empty()) {
        fprintf(stderr, "Error: text encoder output is empty\n");
        return std::vector<float>(text_embed_dim_, 0.0f);
    }

    ncnn::Mat result_mat;
    if (model_type_ == "clip") {
        result_mat = normalize(embeddings);
    } else {
        ncnn::Mat pooled = mean_pool(embeddings);
        result_mat = normalize(pooled);
    }

    std::vector<float> result(text_embed_dim_);
    memcpy(result.data(), result_mat.data, text_embed_dim_ * sizeof(float));

    return result;
}

std::vector<std::vector<float>> ncnn_embedding::encode_text_batch(const std::vector<std::string>& texts) const {
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        results.push_back(encode_text(text));
    }
    return results;
}

std::vector<float> ncnn_embedding::encode_image(const std::vector<unsigned char>& image_data, int width, int height, int channels) const {
    if (!ok_ || !vision_encoder_net) {
        return {};
    }

    ncnn::Mat pixel_values = preprocess_image(image_data, width, height, channels);

    ncnn::Mat embedding;
    {
        ncnn::Extractor ex = vision_encoder_net->create_extractor();
        ex.input("in0", pixel_values);
        ex.extract("out0", embedding);
    }

    if (embedding.empty()) {
        fprintf(stderr, "Error: vision encoder output is empty\n");
        return std::vector<float>(vision_embed_dim_, 0.0f);
    }

    ncnn::Mat normalized = normalize(embedding);

    std::vector<float> result(vision_embed_dim_);
    memcpy(result.data(), normalized.data, vision_embed_dim_ * sizeof(float));

    return result;
}

std::vector<float> ncnn_embedding::encode_image_file(const std::string& image_path) const {
    if (!ok_ || !vision_encoder_net) {
        return {};
    }

    ncnn::Mat raw_image = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(raw_image)) {
        fprintf(stderr, "Error: failed to load image %s\n", image_path.c_str());
        return std::vector<float>(vision_embed_dim_, 0.0f);
    }

    int width = raw_image.w;
    int height = raw_image.h;
    std::vector<unsigned char> image_data(width * height * 3);
    unsigned char* src = (unsigned char*)raw_image.data;
    for (int i = 0; i < width * height; ++i) {
        image_data[i * 3 + 0] = src[i * 3 + 2];
        image_data[i * 3 + 1] = src[i * 3 + 1];
        image_data[i * 3 + 2] = src[i * 3 + 0];
    }

    return encode_image(image_data, width, height, 3);
}