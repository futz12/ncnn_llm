#!/usr/bin/env python3
"""MiniCPM-V 4.7 CPU regression suite, native probes and reference drivers.

One source of truth: no sibling test scripts or checked-in fixture/report files.
CMake emits the readable NATIVE_SOURCES below only into its binary directory.
Python tests are ordinary grouped functions sharing one CLI and probe lookup.
Optional tensor dependencies are loaded only by the selected test group.

Quick start (use a Python environment with the reference dependencies installed):
  cmake -S . -B build -DNCNN_LLM_BUILD_MINICPM_TESTS=ON
  cmake --build build -j2
  python tests/test_minicpm.py smoke --build-dir build
  python tests/test_minicpm.py numerical --build-dir build --model-dir HF_MODEL
  python tests/test_minicpm.py --help

smoke: native image planning/MRoPE, manifest limits, tokenizer metadata, legacy
vision interfaces, synthetic cached language and all vision export tests.
numerical: 1,290 byte-exact Pillow cases, 20 official PIL preprocessing cases,
252 official HF position/RoPE cases, prompt rendering and 6,549 native BPE cases.
cached-decoder: independent HF vs native prefill/decode, explicit/omitted caches.
real: trained vision and decoder parity, then sequential two-image VQA checks
against explicitly provided reference fixtures. Missing inputs/dependencies or
skipped Python tests fail the selected suite; no model downloads are performed.
The repository's original CTest cases remain unchanged and may report their own
optional legacy-checkpoint skips. Direct legacy utility regressions always run.

Experimental scope: original openbmb/MiniCPM-V-4.7-1B, single RGB thumbnail,
16x, CPU FP32. No GPU, quantization, detail slices, video or multi-image claim.
The 'history' command prints the retained measured provenance and limitations.
Reference driver subcommands can recreate fixtures; see each command's --help.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

# Native sources are generated only in the CMake build directory. They compile
# against the same ncnn_llm/image_utils/rope_embed objects as production runtime.
NATIVE_SOURCES = {

    'minicpm_component_probe': r'''
// Synthetic component parity probe. Input/output floats are planar, unpacked FP32.
#include <net.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void read_exact(std::ifstream& f, float* p, size_t n) {
    if (!f.read(reinterpret_cast<char*>(p), n * sizeof(float)))
        throw std::runtime_error("Truncated input tensor");
}

int main(int argc, char** argv) {
    try {
        if (argc != 9) {
            std::cerr << "Usage: probe param bin input output dims width height channels\n";
            return 2;
        }
        int dims = std::stoi(argv[5]), w = std::stoi(argv[6]);
        int h = std::stoi(argv[7]), c = std::stoi(argv[8]);
        if ((dims != 2 && dims != 3) || w <= 0 || h <= 0 || c <= 0)
            throw std::runtime_error("Invalid dimensions");
        ncnn::Mat input;
        if (dims == 2) input.create(w, h);
        else input.create(w, h, c);
        std::ifstream f(argv[3], std::ios::binary);
        if (dims == 2) read_exact(f, input, size_t(w) * h);
        else for (int q = 0; q < c; ++q) read_exact(f, input.channel(q), size_t(w) * h);
        if (f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Excess input data");
        ncnn::Net net;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = false;
        net.opt.use_packing_layout = false;
        net.opt.num_threads = 1;
        if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
            throw std::runtime_error("Failed to load component");
        auto extractor = net.create_extractor();
        ncnn::Mat output;
        if (extractor.input("in0", input) != 0 || extractor.extract("out0", output) != 0)
            throw std::runtime_error("Component inference failed");
        if (output.dims != 2 || output.elempack != 1 || output.elemsize != sizeof(float))
            throw std::runtime_error("Expected unpacked FP32 output matrix");
        std::ofstream result(argv[4], std::ios::binary);
        result.write(reinterpret_cast<const char*>(static_cast<const float*>(output)),
                     size_t(output.w) * output.h * sizeof(float));
        if (!result) throw std::runtime_error("Failed to write output");
        std::cout << output.h << " " << output.w << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
''',

    'minicpm_decoder_probe': r'''
// Synthetic decoder parity probe; never represents trained model acceptance.
#include "kernel/gdr.h"
#include <net.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
using J=nlohmann::json;
static ncnn::Mat read_tensor(const std::filesystem::path& root,const J& spec) {
    auto shape=spec.at("shape").get<std::vector<int>>(); ncnn::Mat m;
    for(int n:shape) if(n<1) throw std::runtime_error("invalid tensor dimension");
    if(shape.size()==1) m.create(shape[0]);
    else if(shape.size()==2) m.create(shape[0],shape[1]);
    else if(shape.size()==3) m.create(shape[0],shape[1],shape[2]);
    else throw std::runtime_error("probe supports dimensions1..3 only");
    if(spec.value("zero",false)) {m.fill(0.f); return m;}
    std::ifstream f(root/spec.at("file").get<std::string>(),std::ios::binary);
    if(m.dims<3) f.read(reinterpret_cast<char*>(m.data),m.w*m.h*sizeof(float));
    else for(int q=0;q<m.c;++q) f.read(reinterpret_cast<char*>(m.channel(q).data),m.w*m.h*sizeof(float));
    if(!f || f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("tensor file length mismatch");
    return m;
}
static J write_tensor(const std::filesystem::path& path,const ncnn::Mat& m) {
    if(m.empty() || m.elempack!=1 || m.elemsize!=4u) throw std::runtime_error("unexpected output precision/packing");
    std::ofstream f(path,std::ios::binary);
    if(m.dims<3) f.write(reinterpret_cast<const char*>(m.data),m.w*m.h*sizeof(float));
    else for(int q=0;q<m.c;++q) f.write(reinterpret_cast<const char*>(m.channel(q).data),m.w*m.h*sizeof(float));
    if(!f) throw std::runtime_error("output write failed");
    J shape=J::array({m.w}); if(m.dims>=2) shape.push_back(m.h); if(m.dims>=3) shape.push_back(m.c);
    return {{"file",path.filename().string()},{"shape",shape}};
}
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::runtime_error("Usage: minicpm_decoder_probe PARAM BIN CASE_JSON OUTPUT_DIR");
        std::ifstream input(argv[3]); J spec; input>>spec;
        const auto root=std::filesystem::path(argv[3]).parent_path(), outdir=std::filesystem::path(argv[4]);
        std::filesystem::create_directories(outdir);
        ncnn::Net net; net.opt.num_threads=2; net.opt.use_packing_layout=false;
        net.opt.use_vulkan_compute=false; net.opt.use_bf16_storage=false;
        net.opt.use_fp16_storage=false; net.opt.use_fp16_arithmetic=false; net.opt.use_fp16_packed=false;
        register_gdr_layers(net,true);
        if(net.load_param(argv[1]) || net.load_model(argv[2])) throw std::runtime_error("decoder load failed");
        std::map<std::string,ncnn::Mat> states;
        for(auto it=spec.at("initial_states").begin();it!=spec.at("initial_states").end();++it)
            states[it.key()]=it.value().is_null()?ncnn::Mat():read_tensor(root,it.value());
        J report=J::array(); int index=0;
        for(const auto& step:spec.at("steps")) {
            auto ex=net.create_extractor();
            for(const auto& [name,value]:states) if(ex.input(name.c_str(),value)) throw std::runtime_error("state input failed: "+name);
            for(auto it=step.at("inputs").begin();it!=step.at("inputs").end();++it)
                if(ex.input(it.key().c_str(),read_tensor(root,it.value()))) throw std::runtime_error("input failed: "+it.key());
            J outputs;
            for(auto it=spec.at("state_outputs").begin();it!=spec.at("state_outputs").end();++it) {
                const auto name=it.value().get<std::string>(); ncnn::Mat result;
                const bool kv=it.key().rfind("cache_k",0)==0 || it.key().rfind("cache_v",0)==0;
                const int rc=ex.extract(name.c_str(),result,kv?1:0);
                if(rc) throw std::runtime_error("state extraction failed: "+name+" rc="+std::to_string(rc));
                // Optimized SDPA KV caches have opaque panel-packed storage;
                // logical w*h bytes include padding and are not tensor data.
                if(kv) outputs[name]={{"shape",J::array({result.w,result.h,result.c})},{"opaque_kvcache",true}};
                else outputs[name]=write_tensor(outdir/(std::to_string(index)+"_"+name+".f32"),result);
                states[it.key()]=result;
            }
            ncnn::Mat result; if(ex.extract("out0",result)) throw std::runtime_error("hidden extraction failed");
            outputs["out0"]=write_tensor(outdir/(std::to_string(index)+"_out0.f32"),result);
            report.push_back(outputs); ++index;
        }
        std::ofstream file(outdir/"outputs.json"); file<<report.dump(2)<<"\n";
        std::cout<<report.dump()<<"\n"; return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n"; return 1;}
}
''',

    'minicpm_preprocess_probe': r'''
#include "utils/image_utils.h"
#include "utils/rope_embed.h"
#include <algorithm>
#include <exception>
extern "C" int minicpm_preprocess_probe(const unsigned char* src,int h,int w,int patch,int scale,float* dst,int capacity,int* shape) {
    try {
        const auto t=ncnn_llm::minicpm::plan_thumbnail(h,w,patch,scale,"16x",true,1);
        shape[0]=t.patch_rows; shape[1]=t.patch_cols;
        auto rgb=ncnn_llm::minicpm::resize_bicubic_rgb(std::vector<uint8_t>(src,src+static_cast<size_t>(h)*w*3),h,w,t.height,t.width);
        auto packed=ncnn_llm::minicpm::pack_rgb(rgb,t.height,t.width,patch);
        if(static_cast<int>(packed.size())>capacity) return -2;
        std::copy(packed.begin(),packed.end(),dst);
        return static_cast<int>(packed.size());
    } catch(const std::exception&) { return -1; }
}
''',

    'minicpm_prompt_probe': r'''
#include "utils/prompt.h"
#include <iostream>
int main() {
    std::string line;
    while(std::getline(std::cin,line)) {
        try {
            auto j=json::parse(line); std::vector<Message> messages;
            for(const auto& m:j.at("messages")) messages.emplace_back(m.at("role"),m.at("content"),m.value("reasoning_content",std::string()));
            std::cout<<json({{"text",apply_chat_template(TemplateType::MINICPM_V47,messages,{},j.value("add_generation_prompt",true),j.value("enable_thinking",false))}}).dump()<<"\n";
        } catch(const std::exception& e) { std::cout<<json({{"error",e.what()}}).dump()<<"\n"; }
    }
}
''',

    'minicpm_resize_probe': r'''
#include "utils/image_utils.h"
#include <algorithm>
#include <exception>
extern "C" int minicpm_resize_probe(const unsigned char* src,int h,int w,int th,int tw,int bgr,unsigned char* dst) {
    try {
        std::vector<uint8_t> input;
        if(h>0 && w>0) input.assign(src,src+static_cast<size_t>(h)*w*3);
        const auto output=ncnn_llm::minicpm::resize_bicubic_rgb(input,h,w,th,tw,bgr!=0);
        std::copy(output.begin(),output.end(),dst);
        return 0;
    } catch(const std::exception&) { return -1; }
}
''',

    'minicpm_tokenizer_probe': r'''
// Standalone actual C++ tokenizer probe; no ncnn/Torch dependencies.
#include "utils/tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
static std::string unhex(const std::string& value) {
    std::string out;
    for(size_t i=0;i<value.size();i+=2) out.push_back(static_cast<char>(std::stoul(value.substr(i,2),nullptr,16)));
    return out;
}
static std::string hex(const std::string& value) {
    const char* digits="0123456789abcdef";std::string out;
    for(unsigned char c:value) {out+=digits[c>>4];out+=digits[c&15];}return out;
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: probe vocab merges artifact");
        SpecialTokensConfig spec;
        auto tokenizer=BpeTokenizer::LoadFromFiles(argv[1],argv[2],spec,false,false,true);
        tokenizer.LoadMiniCPMTokenizer(argv[3]);
        // Exercise move construction and assignment after enabling the artifact.
        auto moved=std::move(tokenizer);
        tokenizer=std::move(moved);
        std::string line;
        while(std::getline(std::cin,line)) {
            try {
                auto ids=tokenizer.encode(unhex(line),false,false);
                for(size_t i=0;i<ids.size();++i) {if(i)std::cout<<',';std::cout<<ids[i];}
                std::cout<<'|'<<hex(tokenizer.decode(ids,false))<<'|'<<hex(tokenizer.decode(ids,true))<<'\n';
            } catch(const std::exception& error) {std::cout<<"ERROR:"<<error.what()<<'\n';}
        }
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
''',

    'minicpm_trained_vision_probe': r'''
// Real-checkpoint vision-only parity probe; never allocates the language model.
#include "ncnn_llm_gpt.h"
#include "utils/image_utils.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "Usage: minicpm_trained_vision_probe MODEL_DIR IMAGE OUTPUT_F32 [THREADS=2]\n";
            return 2;
        }
        const std::string directory = argv[1];
        const int threads = argc == 5 ? std::stoi(argv[4]) : 2;
        if (threads < 1 || threads > 64) throw std::runtime_error("Invalid thread count");
        std::ifstream config_file(directory + "/model.json");
        if (!config_file) throw std::runtime_error("Cannot read model.json");
        nlohmann::json config;
        config_file >> config;
        ncnn::Mat image = load_image_to_ncnn_mat(argv[2]);
        if (image.empty()) throw std::runtime_error("Cannot decode image");
        std::cerr << "vision_load_start\n";
        auto vision = ncnn_llm_gpt::load_minicpm_vision(directory, config.at("setting").at("vision"), threads);
        std::cerr << "vision_load_complete\n";
        ncnn_llm::minicpm::Thumbnail thumbnail{};
        std::cerr << "vision_encode_start\n";
        const ncnn::Mat output = ncnn_llm_gpt::encode_minicpm_vision(*vision, image, thumbnail);
        std::cerr << "vision_encode_complete\n";
        std::ofstream file(argv[3], std::ios::binary);
        for (int row = 0; row < output.h; ++row)
            file.write(reinterpret_cast<const char*>(output.row(row)), output.w * sizeof(float));
        if (!file) throw std::runtime_error("Cannot write feature output");
        std::cout << nlohmann::json({{"rows", output.h}, {"cols", output.w},
            {"patch_rows", thumbnail.patch_rows}, {"patch_cols", thumbnail.patch_cols},
            {"threads", threads}, {"language_model_loaded", false}}).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
''',

    'minicpm_vision_reference_probe': r'''
// C++ bridge for the consolidated positions reference group. No model weights.
#include "utils/image_utils.h"
#include "utils/rope_embed.h"
#include <iomanip>
#include <iostream>
using namespace ncnn_llm::minicpm;
int main() {
    std::cout << std::setprecision(9);
    std::string mode;
    int h,w,side,dim,n;
    int64_t base;
    double theta;
    MarkerIds m;
    std::array<int,3> sections;
    while(std::cin >> mode >> h >> w >> side >> base >> dim >> theta
          >> m.image >> m.image_start >> m.image_end >> m.slice_start >> m.slice_end >> m.newline
          >> sections[0] >> sections[1] >> sections[2] >> n) {
        std::vector<int> ids(n); for(auto& id:ids) std::cin >> id;
        auto p=canvas_positions(ids,m,h,w,mode,base);
        std::vector<float> cc,ss; interleaved_rope(p,dim,theta,sections,cc,ss);
        std::cout << p.rope_delta << ' ' << p.next_position << '\n';
        for(auto xyz:p.positions) { for(auto x:xyz) std::cout << x << ' '; }
        std::cout << '\n';
        for(auto x:nearest_position_ids(h,w,side)) std::cout << x << ' ';
        std::cout << '\n';
        for(auto x:cc) std::cout << x << ' ';
        std::cout << '\n';
        for(auto x:ss) std::cout << x << ' ';
        std::cout << '\n';
    }
}
''',

    'test_minicpm_runtime': r'''
// Real-checkpoint smoke/parity runner. Not registered with CTest: weights are optional.
#include "ncnn_llm_gpt.h"
#include "utils/prompt.h"
#include "utils/minicpm_tokenizer_config.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--check-limits") {
            ncnn_llm_gpt::validate_minicpm_thumbnail({896,896,64,64,16,16});
            for(auto shape : std::vector<std::pair<int,int>>{{1,4097},{0,64},{-1,64},{2147483647,2147483647}}) {
                bool rejected=false;
                try { ncnn_llm_gpt::validate_minicpm_thumbnail({0,0,shape.first,shape.second,0,0}); }
                catch(const std::runtime_error&) { rejected=true; }
                if(!rejected) throw std::runtime_error("thumbnail limit boundary was accepted");
            }
            const auto extreme=ncnn_llm::minicpm::plan_thumbnail(1,16384,14,448,"16x",false,1);
            bool rejected=false;
            try { ncnn_llm_gpt::validate_minicpm_thumbnail(extreme); }
            catch(const std::runtime_error&) { rejected=true; }
            if(!rejected) throw std::runtime_error("extreme-aspect thumbnail was accepted");
            nlohmann::json manifest={{"type","minicpm_v47"},{"format_version",1},{"weight_dtype","float32"},
                {"scope","single_image_thumbnail"},{"downsample_mode","16x"},{"hidden_size",1152},{"output_hidden_size",1024},
                {"position_grid_size",70},{"patch_size",14},{"window_kernel_size",{2,2}},{"merge_kernel_size",{2,2}},
                {"num_hidden_layers",27},{"insert_layer_id",6},{"position_embeddings",{{"rows",4900},{"cols",1152}}}};
            ncnn_llm_gpt::validate_minicpm_vision_manifest(manifest);
            for(auto bad : std::vector<nlohmann::json>{{{"format_version",2}},{{"weight_dtype","bfloat16"}},
                    {{"hidden_size",2147483647}},{{"position_grid_size",2147483647}},{{"window_kernel_size",{2,3}}},
                    {{"insert_layer_id",27}},{{"num_hidden_layers",0}},{{"position_embeddings",{{"rows",4899},{"cols",1152}}}}}) {
                auto changed=manifest; changed.update(bad); bool rejected=false;
                try { ncnn_llm_gpt::validate_minicpm_vision_manifest(changed); }
                catch(const std::exception&) { rejected=true; }
                if(!rejected) throw std::runtime_error("invalid manifest boundary was accepted");
            }
            ncnn_llm_gpt_base_ctx base; base.image_count=1;
            qwen3_5_ctx hybrid; hybrid.image_count=1;
            if(base.clone()->image_count!=1 || hybrid.clone()->image_count!=1)
                throw std::runtime_error("context clone dropped image count");
            std::cout<<"Thumbnail, manifest and context-clone limits passed\n"; return 0;
        }
        if(argc==4 && std::string(argv[1])=="--tokenize") {
            const std::string dir=argv[2];
            std::ifstream config_file(dir+"/model.json"), reference_file(argv[3]);
            nlohmann::json config, reference; config_file>>config; reference_file>>reference;
            const auto& tc=config.at("tokenizer");
            auto tokenizer=BpeTokenizer::LoadFromFiles(dir+"/"+tc.at("vocab_file").get<std::string>(),
                dir+"/"+tc.at("merges_file").get<std::string>(),SpecialTokensConfig{},false,true,tc.value("type",std::string("bpe"))=="bbpe");
            for(const auto& token:tc.at("additional_special_tokens")) tokenizer.AddAdditionalSpecialToken(token.get<std::string>());
            const bool is_minicpm=config.at("setting").at("vision").value("type",std::string())=="minicpm_v4_7";
            const auto artifact=ncnn_llm::minicpm_tokenizer_artifact(dir,tc,is_minicpm);
            if(is_minicpm) tokenizer.LoadMiniCPMTokenizer(artifact);
            auto ids=tokenizer.encode(reference.at("text").get<std::string>(),false,false);
            const auto bos=tc.at("bos").get<std::string>();
            if(!bos.empty() && config.at("setting").at("vision").value("type",std::string())!="minicpm_v4_7") ids.insert(ids.begin(),tokenizer.token_to_id().at(bos));
            const auto expected=reference.at("ids").get<std::vector<int>>();
            std::cout<<nlohmann::json({{"ids",ids},{"matched",ids==expected},{"count",ids.size()}}).dump()<<"\n";
            return ids==expected?0:1;
        }
        if(argc<4) { std::cerr<<"Usage: test_minicpm_runtime MODEL_DIR IMAGE PROMPT [MAX_TOKENS=32] [VISION_F32_DUMP]\n       test_minicpm_runtime --tokenize MODEL_DIR HF_REFERENCE_JSON\n"; return 2; }
        const std::string dir=argv[1]; auto image=load_image_to_ncnn_mat(argv[2]);
        if(ncnn_mat_empty(image)) throw std::runtime_error("cannot load image");
        std::ifstream file(dir+"/model.json"); nlohmann::json config; file>>config;
        if(argc>5) {
            auto vision = ncnn_llm_gpt::load_minicpm_vision(dir,config.at("setting").at("vision"),4);
            ncnn_llm::minicpm::Thumbnail thumb{}; auto features=ncnn_llm_gpt::encode_minicpm_vision(*vision, image,thumb);
            std::ofstream dump(argv[5],std::ios::binary);
            for(int r=0;r<features.h;++r) dump.write(reinterpret_cast<const char*>(features.row(r)),features.w*sizeof(float));
            if(!dump) throw std::runtime_error("cannot write vision dump");
            std::cerr<<"vision_shape="<<features.h<<","<<features.w<<"\n";
        }
        ncnn_llm_gpt model(dir,false,4,0,false,true);
        const std::vector<Message> messages={{"system","You are a helpful assistant."},
            {"user","<image>\n"+std::string(argv[3])}};
        const auto full=apply_chat_template(TemplateType::MINICPM_V47,messages,{},true,false);
        const auto suffix=apply_chat_template(TemplateType::MINICPM_V47,{messages.back()},{},true,false);
        auto ctx=model.prefill(full.substr(0,full.size()-suffix.size()));
        ctx=model.prefill(suffix,image,ctx);
        std::cerr<<"prefill_token="<<ctx->cur_token<<" next_position="<<ctx->position_id<<"\n";
        GenerateConfig cfg; cfg.max_new_tokens=argc>4?std::stoi(argv[4]):32; cfg.do_sample=0;
        cfg.temperature=0; cfg.repetition_penalty=1; cfg.top_k=0; cfg.top_p=1;
        model.generate(ctx,cfg,[](const std::string& text){ std::cout<<text<<std::flush; });
        std::cout<<"\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
''',

    'test_minicpm_tokenizer_config': r'''
#include "utils/minicpm_tokenizer_config.h"
#include <fstream>
#include <iostream>
#include <chrono>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    namespace fs=std::filesystem;
    fs::create_directories(argv[1]);
    const fs::path root=fs::path(argv[1])/("case-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if(!fs::create_directory(root)) throw std::runtime_error("cannot create fresh fixture directory");
    struct Cleanup {fs::path path; ~Cleanup(){std::error_code ec;fs::remove_all(path,ec);}} cleanup{root};
    std::ofstream(root/"artifact.txt")<<"test";
    auto valid=nlohmann::json{{"pretokenizer",{{"type","minicpm_v47"},{"file","artifact.txt"}}}};
    int checked=0;
    auto reject=[&](const nlohmann::json& config,bool mini) {
        try { ncnn_llm::minicpm_tokenizer_artifact(root.string(),config,mini); }
        catch(const std::exception&) { ++checked; return; }
        throw std::runtime_error("invalid tokenizer configuration accepted");
    };
    if(ncnn_llm::minicpm_tokenizer_artifact(root.string(),valid,true)!=(fs::canonical(root)/"artifact.txt").string()) return 1;
    if(!ncnn_llm::minicpm_tokenizer_artifact(root.string(),nlohmann::json::object(),false).empty()) return 1;
    reject(nlohmann::json::object(),true); reject(valid,false);
    auto bad=valid; bad["pretokenizer"]["type"]="legacy"; reject(bad,true);
    for(const auto& name:{"../artifact.txt","/tmp/escape","a\\b","a:b","missing","./artifact.txt",""}) {
        bad=valid; bad["pretokenizer"]["file"]=name; reject(bad,true);
    }
    std::error_code symlink_error;
    fs::create_symlink("artifact.txt",root/"link",symlink_error);
    if(!symlink_error) {
        bad=valid; bad["pretokenizer"]["file"]="link"; reject(bad,true);
    } else {
#ifdef _WIN32
        std::cout<<"Symlink case skipped: OS did not allow fixture creation\n";
#else
        throw fs::filesystem_error("cannot create symlink fixture",root/"link",symlink_error);
#endif
    }
    fs::resize_file(root/"artifact.txt",1024*1024+1); reject(valid,true);
    std::cout<<"Tokenizer configuration checks passed: 2 accepted, "<<checked<<" rejected\n";
}
''',

    'test_minicpm_vision': r'''
// Build and run with CMake/CTest; exercises the shared image and RoPE utilities.
#include "utils/image_utils.h"
#include "utils/rope_embed.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace ncnn_llm::minicpm;
// Keep numerical and rejection checks active in Release builds.
#define CHECK(...) do { if (!(__VA_ARGS__)) throw std::runtime_error("check failed: " #__VA_ARGS__); } while (false)
template<class F> void rejects(F f) { bool caught=false; try {f();} catch(const std::invalid_argument&) {caught=true;} CHECK(caught); }
int main() {
    auto square=plan_thumbnail(100,100,14,448,"16x");
    CHECK(square.height==448 && square.width==448 && square.token_count()==64);
    auto wide=plan_thumbnail(300,400,14,448,"4x");
    CHECK(wide.height==392 && wide.width==504 && wide.token_count()==252);
    auto portrait=plan_thumbnail(400,300,14,448,"16x");
    CHECK(portrait.height==504 && portrait.width==392 && portrait.token_count()==63);
    // Python round tie-to-even: 140 / 56 = 2.5 => 112.
    auto tie=plan_thumbnail(140,140,14,140,"16x"); CHECK(tie.height==112);
    rejects([] {plan_thumbnail(1000,1000,14,448,"16x");});
    CHECK(plan_thumbnail(1000,1000,14,448,"16x",false).token_count()==64);
    CHECK(plan_thumbnail(1000,1000,14,448,"16x",true,1).token_count()==64);
    rejects([] {plan_thumbnail(0,1,14,448,"16x");});
    rejects([] {plan_thumbnail(1,1,14,448,"8x");});
    std::vector<uint8_t> rgb(4*4*3);
    for(size_t i=0;i<rgb.size();++i) rgb[i]=static_cast<uint8_t>(i);
    auto packed=pack_rgb(rgb,4,4,2);
    // c=0, local patch row=0: global pixels 0,1,2,3,8,9,10,11.
    CHECK(std::abs(packed[4]-((24.f/255.f-.5f)/.5f))<1e-7);
    CHECK(std::abs(packed[8]-((12.f/255.f-.5f)/.5f))<1e-7);
    CHECK(std::abs(packed[16]-((1.f/255.f-.5f)/.5f))<1e-7);
    rejects([&] {pack_rgb(rgb,3,4,2);});
    CHECK((nearest_position_ids(2,3,4)==std::vector<int>{0,1,2,8,9,10}));
    const MarkerIds m={100,101,102,103,104,105};
    std::vector<int> ids={7,8,101,100,100,100,100,100,100,102,9};
    auto pos=canvas_positions(ids,m,8,12,"16x");
    CHECK((pos.positions[2]==std::array<int64_t,3>{{2,1,1}}));
    CHECK((pos.positions[3]==std::array<int64_t,3>{{2,2,2}}));
    CHECK((pos.positions[8]==std::array<int64_t,3>{{2,3,4}}));
    CHECK((pos.positions[9]==std::array<int64_t,3>{{2,4,5}}));
    CHECK((pos.positions[10]==std::array<int64_t,3>{{6,6,6}}));
    CHECK(pos.next_position==7 && pos.rope_delta==-4);
    CHECK(canvas_positions(ids,m,4,6,"4x").positions==pos.positions);
    auto offset=canvas_positions(ids,m,8,12,"16x",20);
    CHECK(offset.next_position==27 && offset.positions[2][1]==21);
    std::vector<int> no_prefix={101,100,102,105,105,9};
    auto zero=canvas_positions(no_prefix,m,4,4,"16x");
    CHECK((zero.positions[0]==std::array<int64_t,3>{{0,0,0}}));
    CHECK(zero.positions[2][1]==1 && zero.positions[3][1]==0 && zero.positions[5][1]==2);
    CHECK(zero.next_position==3 && zero.rope_delta==-3);
    rejects([&] {canvas_positions(ids,m,8,12,"4x");});
    auto bad=ids; bad.push_back(103); rejects([&] {canvas_positions(bad,m,8,12,"16x");});
    bad=ids; bad[0]=100; rejects([&] {canvas_positions(bad,m,8,12,"16x");});
    bad=ids; bad.push_back(101); rejects([&] {canvas_positions(bad,m,8,12,"16x");});
    std::vector<float> cc,ss;
    interleaved_rope(pos,12,10000,{{2,2,2}},cc,ss);
    CHECK(cc.size()==ids.size()*6 && ss.size()==cc.size());
    CHECK(std::abs(cc[2*6+1]-std::cos(1.0/std::pow(10000.,2./12)))<1e-6);
    rejects([&] {interleaved_rope(pos,12,10000,{{1,1,1}},cc,ss);});
    std::cout << "MiniCPM thumbnail, packing, canvas and RoPE tests passed\n";
}
''',

    'test_vision_utils': r'''
// Regression coverage for the shared legacy and pre-expanded image paths.
#include "utils/image_utils.h"
#include "utils/rope_embed.h"
#include "utils/vision_rope.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
static void injection(bool expanded, bool at_end) {
    const int pad = 151655; // Must be an integer fill, not byte-wise memset.
    std::vector<int> ids = expanded ? std::vector<int>{11, pad, pad} : std::vector<int>{11, pad};
    if (!at_end) ids.push_back(22);
    ncnn::Mat text(3, (int)ids.size()), image(3, 2);
    for (int y=0; y<text.h; ++y) for (int x=0; x<3; ++x) text.row(y)[x]=y*10+x;
    for (int y=0; y<2; ++y) for (int x=0; x<3; ++x) image.row(y)[x]=100+y*10+x;
    const int old_last = (text.h-1)*10;
    int offset=-1;
    inject_image_embeds(ids, text, offset, pad, image, expanded);
    check(offset==1 && text.h==(at_end ? 3:4), "injected shape or offset");
    check(ids== (at_end ? std::vector<int>{11,pad,pad} : std::vector<int>{11,pad,pad,22}), "image token ids");
    for (int x=0; x<3; ++x) {
        check(text.row(0)[x]==x, "prefix preserved");
        check(text.row(1)[x]==100+x && text.row(2)[x]==110+x, "image rows copied");
        if (!at_end) check(text.row(3)[x]==old_last+x, "suffix preserved");
    }
}
int main() {
    try {
        for (bool expanded : {false,true}) for (bool at_end : {false,true}) injection(expanded,at_end);
        ncnn::Mat text(2,3), image(2,2); text.fill(7.f); image.fill(42.f);
        std::vector<int> malformed{1,99,2}; int index=12;
        inject_image_embeds(malformed,text,index,99,image,true);
        check(index==-1 && text.row(1)[0]==7.f && malformed.size()==3, "invalid expanded span rejected atomically");
        std::vector<int> absent{1,2,3};
        inject_image_embeds(absent,text,index,99,image);
        check(index==-1 && text.row(1)[0]==7.f, "missing legacy placeholder unchanged");
        ncnn::Mat bgr(4,4,3,(size_t)1u);
        for (int i=0;i<48;++i) static_cast<unsigned char*>(bgr.data)[i]=(i*37)%256;
        auto bilinear=ncnn_mat_resize(bgr,4,4);
        auto bicubic=ncnn_mat_resize_bicubic(bgr,4,4);
        check(!bilinear.empty() && !bicubic.empty(), "legacy resize outputs");
        check(std::memcmp(bgr.data,bilinear.data,48)==0, "legacy bilinear identity");
        check(std::memcmp(bgr.data,bicubic.data,48)==0, "legacy bicubic identity");
        check(ncnn_mat_resize({},4,4).empty() && ncnn_mat_resize_bicubic({},4,4).empty(), "empty legacy resize");
        ncnn::Mat cc, ss;
        generate_vision_rope_cache_2d(4,4,2,10000.f,{2,2},true,cc,ss);
        check(cc.w==8 && cc.h==16 && ss.w==8 && ss.h==16, "legacy vision rope shape");
        int row=0;
        for(int gy=0;gy<2;++gy) for(int gx=0;gx<2;++gx)
            for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx,++row) {
                for(int j=0;j<4;++j) {
                    float angle=(j<2?gy*2+dy:gx*2+dx)*(j%2?0.01f:1.f);
                    check(std::abs(cc.row(row)[j]-std::cos(angle))<1e-6f &&
                          std::abs(ss.row(row)[j]-std::sin(angle))<1e-6f, "legacy rope ordering");
                    check(cc.row(row)[j]==cc.row(row)[j+4] && ss.row(row)[j]==ss.row(row)[j+4], "legacy rope duplication");
                }
            }
        std::cout << "PASS: legacy/expanded injection, legacy image resizing and vision RoPE\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
''',
}



def window_indices(height, width):
    """Flatten row-major 2x2 windows, in row-major window order."""
    import numpy as np
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("Patch grid must be positive and divisible by 2")
    return np.arange(height * width).reshape(height // 2, 2, width // 2, 2).transpose(0, 2, 1, 3).reshape(-1)


def nearest_position_ids(height, width, position_grid_size=70):
    """SigLIP nearest-grid buckets (not bilinear interpolation)."""
    import numpy as np
    if min(height, width, position_grid_size) <= 0:
        raise ValueError("Grid dimensions must be positive")
    boundaries = np.arange(1, position_grid_size, dtype=np.float32) / position_grid_size
    ys = np.arange(height, dtype=np.float32) / height
    xs = np.arange(width, dtype=np.float32) / width
    rows = np.searchsorted(boundaries, ys, side="right")
    cols = np.searchsorted(boundaries, xs, side="right")
    return (rows[:, None] * position_grid_size + cols[None, :]).reshape(-1)



class ReadOnlyVisionPackage:
    """Load existing graphs and verify that validation never changes their bytes."""
    def __init__(self, directory, prefix=""):
        self.directory = Path(directory).resolve()
        model = self.directory / 'model.json'
        name = (json.loads(model.read_text())['setting']['vision']['manifest']
                if model.is_file() and not prefix else prefix + 'minicpm_v47_vision.json')
        manifest_path = self.path(name)
        self.manifest = json.loads(manifest_path.read_text())
        paths = [manifest_path, self.path(self.manifest['position_embeddings']['file'])]
        if model.is_file():
            paths.append(model)
        for component in self.manifest['components'].values():
            paths.extend(self.path(component[key]) for key in ('param', 'bin'))
        self.paths = sorted(set(paths))
        self.before = self.fingerprints()

    def path(self, name):
        path = (self.directory / name).resolve()
        if self.directory not in path.parents or not path.is_file():
            raise ValueError(f'Missing or external converted artifact: {name}')
        return path

    def fingerprints(self):
        import hashlib
        result = {}
        for path in self.paths:
            digest = hashlib.sha256()
            with path.open('rb') as file:
                for block in iter(lambda: file.read(1024 * 1024), b''):
                    digest.update(block)
            result[str(path.relative_to(self.directory))] = digest.hexdigest()
        return result

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        after = self.fingerprints()
        if after != self.before:
            raise AssertionError('Validation modified converted model artifacts')


def evaluate_vision_package(package, pixels, threads=2, on_stage=None):
    """Read-only ncnn forward; shared by real parity and corruption regression."""
    import gc
    import ncnn
    import numpy as np
    manifest = package.manifest
    width = manifest['hidden_size']
    height, columns = (int(n // manifest['patch_size']) for n in pixels.shape[1:])

    def load(name):
        net = ncnn.Net()
        net.opt.num_threads = threads
        net.opt.use_vulkan_compute = False
        net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = net.opt.use_fp16_packed = False
        net.opt.use_bf16_storage = net.opt.use_packing_layout = False
        item = manifest['components'][name]
        assert net.load_param(str(package.path(item['param']))) == 0
        assert net.load_model(str(package.path(item['bin']))) == 0
        return net

    def forward(net, values):
        extractor = net.create_extractor()
        owner = np.ascontiguousarray(values, dtype=np.float32)
        tensor = ncnn.Mat(owner).clone()
        assert extractor.input('in0', tensor) == 0
        status, output = extractor.extract('out0')
        assert status == 0
        return np.array(output).copy()

    def run(name, values):
        net = load(name)
        output = forward(net, values)
        del net
        gc.collect()
        return output

    def stage(name, values):
        if on_stage:
            on_stage(name, values)
        return values

    x = run('patch_embed', pixels)
    pos = manifest['position_embeddings']
    position = np.fromfile(package.path(pos['file']), dtype='<f4').reshape(pos['rows'], pos['cols'])
    x += position[nearest_position_ids(height, columns, manifest['position_grid_size'])]
    del position
    stage('embedding', x)
    x = stage('pre', run('encoder_pre', x))
    windows = x[window_indices(height, columns)].reshape(-1, 4, width)
    net = load('window_attn')
    attended = np.stack([forward(net, window) for window in windows])
    del net, windows
    x = stage('window', run('window_mlp', attended.reshape(-1, width * 4)) + attended.mean(axis=1))
    del attended
    x = stage('post', run('encoder_post', x))
    x = x[window_indices(height // 2, columns // 2)].reshape(-1, width * 4)
    return stage('features', run('projector', x))


def _check_minicpm_v47_vision_parity():
    """Actual trained single-thumbnail HF-to-ncnn visual-feature parity (CPU FP32)."""
    import argparse, gc, json, os, sys, time
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from pathlib import Path
    import numpy as np
    import ncnn
    import torch
    from PIL import Image
    from safetensors import safe_open

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("image", type=Path, help="448x448 RGB photo; preprocessing is verified separately")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("converted_dir", type=Path)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--min-cosine", type=float, default=.999999)
    args = parser.parse_args()
    ROOT, OUT, CONVERTED, PREFIX = args.model_dir, args.output_dir, args.converted_dir, args.prefix

    if OUT.resolve() == CONVERTED.resolve() or CONVERTED.resolve() in OUT.resolve().parents:
        raise ValueError('Reference output must not be inside the converted model package')
    with ReadOnlyVisionPackage(CONVERTED, PREFIX) as package:
        OUT.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(args.threads)
        provenance_path = ROOT/'download-manifest.json'
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        entry = next((f for f in provenance.get('files', []) if f['name']=='model.safetensors'), None)
        assert (ROOT/'model.safetensors').exists(), 'A complete local safetensors checkpoint is required'
        if entry is not None:
            assert (ROOT/'model.safetensors').stat().st_size == entry['bytes'], 'Verified checkpoint size mismatch'
        else:
            entry = {'sha256': None}

        image = np.asarray(Image.open(args.image).convert('RGB'))
        assert image.shape == (448,448,3)
        pixels = image.transpose(2,0,1).astype(np.float32)/127.5 - 1
        # Standard RGB CHW produces the same patch row order as processor NaViT packing.
        np.save(OUT/'pixels_chw.npy', pixels)
        targets=torch.tensor([[32,32]], dtype=torch.int32)
        from transformers.models.minicpmv4_7.configuration_minicpmv4_7 import MiniCPMV4_7Config
        from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7Model
        cfg=MiniCPMV4_7Config.from_pretrained(ROOT,local_files_only=True)
        cfg._attn_implementation='eager';cfg.vision_config._attn_implementation='eager'
        print('Constructing meta reference; language tensors remain meta',flush=True)
        with torch.device('meta'):
            model=MiniCPMV4_7Model(cfg)
        with safe_open(str(ROOT/'model.safetensors'),framework='pt',device='cpu') as sf:
            count=0
            for key in sf.keys():
                if not key.startswith(('model.vision_tower.','model.merger.')):
                    continue
                relative=key.removeprefix('model.')
                parent,name=relative.rsplit('.',1)
                module=model.get_submodule(parent)
                assert name in module._parameters,(key,'not a parameter')
                module._parameters[name]=torch.nn.Parameter(sf.get_tensor(key).float(),requires_grad=False)
                count+=1
        print('Loaded vision parameters',count,flush=True)
        assert all(not p.is_meta for p in model.vision_tower.parameters())
        assert all(not p.is_meta for p in model.merger.parameters())
        assert all(p.is_meta for p in model.language_model.parameters())
        assert not any(b.is_meta for b in model.vision_tower.buffers()),'Unexpected vision meta buffers'
        model.eval()
        def hook(name):
            def save(_module,_inputs,result):
                if isinstance(result,tuple):result=result[0]
                np.save(OUT/(name+'.npy'),result.detach().float().cpu().numpy().squeeze(0))
            return save
        handles=[]
        for name,mod in [('hf_embedding',model.vision_tower.embeddings),('hf_pre',model.vision_tower.encoder.layers[6]),('hf_window',model.vision_tower.vit_merger),('hf_post',model.vision_tower.post_layernorm)]:
            handles.append(mod.register_forward_hook(hook(name)))
        start=time.time()
        with torch.no_grad():
            features=model.get_image_features(torch.from_numpy(pixels).unsqueeze(0),targets,downsample_mode='16x').pooler_output[0]
            np.save(OUT/'hf_features.npy',features.numpy())
        print('HF vision seconds',time.time()-start,'shape',tuple(features.shape),flush=True)
        for h in handles:h.remove()
        del handles, model, features, module, mod
        # Ensure references from construction/loading loops don't retain large tensors.
        gc.collect()
        print('HF reference released; validating existing converted vision components',flush=True)
        reports={}
        def compare(name,actual):
            expected=np.load(OUT/('hf_'+name+'.npy'))
            np.save(OUT/('ncnn_'+name+'.npy'),actual)
            assert actual.shape==expected.shape,(name,actual.shape,expected.shape)
            a,b=actual.astype(np.float64).ravel(),expected.astype(np.float64).ravel()
            assert np.isfinite(a).all() and np.isfinite(b).all(), (name, 'nonfinite output')
            delta=np.abs(a-b)
            reports[name]={'shape':list(actual.shape),'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),'cosine_similarity':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))}
            print(name,reports[name],flush=True)
            np.testing.assert_allclose(actual, expected, atol=args.atol, rtol=args.rtol)
            assert reports[name]['cosine_similarity'] > args.min_cosine, reports[name]
        evaluate_vision_package(package, pixels, args.threads, compare)
        reports['metadata']={'checkpoint_sha256':entry['sha256'],'revision':provenance.get('revision'),'ncnn_version':getattr(ncnn,'__version__','unknown'),'image':str(args.image),'dtype':'float32','pixel_normalization':'RGB/127.5-1','hf_vision_only':True,'acceptance':{'atol':args.atol,'rtol':args.rtol,'min_cosine':args.min_cosine},'input_buffer_retained':True,'converted_sha256':package.before}
        (OUT/'parity.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('PASS trained visual feature absolute/relative and cosine checks',flush=True)
    return types.SimpleNamespace(**locals())


def compare_cached_decoder(root, output_name="native"):
    import json
    from pathlib import Path
    import sys
    import numpy as np
    root=Path(root); native=root/output_name
    references=json.loads((root/'references.json').read_text())
    outputs=json.loads((native/'outputs.json').read_text())
    if len(outputs)!=3 or len(references)!=3:
        raise AssertionError("all three prefill/decode steps are required")
    report=[]
    for index,result in enumerate(outputs):
        for name,expected in references[str(index)].items():
            actual=result[name]
            if actual['shape']!=expected['shape']:
                raise AssertionError((index,name,actual['shape'],expected['shape']))
            a=np.fromfile(native/actual['file'],dtype='<f4')
            b=np.fromfile(root/expected['file'],dtype='<f4')
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise AssertionError((index,name,"nonfinite output"))
            np.testing.assert_allclose(a,b,atol=5e-5,rtol=5e-5,err_msg=f'step{index}/{name}')
            report.append({'step':index,'name':name,'max_abs':float(np.max(np.abs(a-b)))})
        for name in ('out_cache_k0','out_cache_v0'):
            if result[name]['shape'][1]!=(8,9,12)[index]:
                raise AssertionError((index,name,result[name]['shape']))
    output={'synthetic':True,'trained_model':False,'native_parity':report}
    (root/(output_name+'-parity-report.json')).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
    return output


def _compare_minicpm_decoder_probe():
    return types.SimpleNamespace(main=lambda: compare_cached_decoder(sys.argv[1]))


def _compare_minicpm_trained_decoder():
    """Compare a trained native decoder probe to a saved independent HF target.
    Usage: python TEST FIXTURE_DIR NATIVE_OUTPUT_DIR
    This checks decoder hidden states only, not complete image-question inference.
    """
    import json
    from pathlib import Path
    import sys
    import numpy as np

    root, output = map(Path, sys.argv[1:3])
    report = json.loads((output/'outputs.json').read_text())
    assert len(report)==1, 'expected one trained eight-token prefill'
    spec=report[0]['out0']
    assert spec['shape']==[1024,8], spec
    actual=np.fromfile(output/spec['file'],dtype='<f4').astype(np.float64)
    expected=np.fromfile(root/'target.f32',dtype='<f4').astype(np.float64)
    assert actual.shape==expected.shape==(8192,)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    diff=np.abs(actual-expected)
    cosine=float(np.dot(actual,expected)/(np.linalg.norm(actual)*np.linalg.norm(expected)))
    metrics={'trained_decoder':True,'full_vqa':False,'shape':spec['shape'],
             'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'cosine':cosine,
             'relative_l2':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
             'atol':5e-4,'rtol':5e-4,
             'passed':bool(np.allclose(actual,expected,atol=5e-4,rtol=5e-4) and cosine>=0.99999)}
    (output/'trained-parity.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
    if not metrics['passed']: raise SystemExit(1)
    return types.SimpleNamespace(**locals())


def _streamed_reference():
    """Memory-bounded *official-model test oracle* tensor adapters.

    Only embedding lookup and the final bias-free linear projection are streamed.
    Decoder, visual-token scatter, M-RoPE and caches remain official Transformers.
    This is not an ncnn runtime or alternate inference engine.
    """
    from pathlib import Path

    import torch
    from safetensors import safe_open


    def locate_tensor(model_dir, key):
        matches = []
        for path in sorted(Path(model_dir).glob('*.safetensors')):
            with safe_open(str(path), framework='pt', device='cpu') as checkpoint:
                if key in checkpoint.keys():
                    matches.append((str(path), tuple(checkpoint.get_slice(key).get_shape())))
        if len(matches) != 1:
            raise ValueError(f'expected one checkpoint tensor {key}, found {len(matches)}')
        path, shape = matches[0]
        if len(shape) != 2:
            raise ValueError(f'expected a matrix for {key}, found {shape}')
        return path, shape


    class StreamedEmbedding(torch.nn.Module):
        """Exact selected rows of checkpoint embeddings, promoted to CPU FP32."""
        def __init__(self, model_dir, key='model.language_model.embed_tokens.weight'):
            super().__init__()
            self.path, shape = locate_tensor(model_dir, key)
            self.key = key
            self.num_embeddings, self.embedding_dim = shape
            self.padding_idx = None  # Preserve every checkpoint row, including PAD.

        @torch.no_grad()
        def forward(self, input_ids):
            if input_ids.device.type != 'cpu' or input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError('streamed reference embedding requires CPU integer IDs')
            flat = input_ids.reshape(-1)
            if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= self.num_embeddings):
                raise IndexError('token ID outside checkpoint vocabulary')
            unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
            selected = torch.empty((unique.numel(), self.embedding_dim), dtype=torch.float32)
            with safe_open(self.path, framework='pt', device='cpu') as checkpoint:
                source = checkpoint.get_slice(self.key)
                for i, token in enumerate(unique.tolist()):
                    selected[i].copy_(source[token:token+1, :].reshape(-1).float())
            return selected[inverse].reshape(*input_ids.shape, self.embedding_dim)


    class StreamedLMHead(torch.nn.Module):
        """Same FP32 linear map as nn.Linear; vocabulary rows bounded by chunk_rows.

        Full logits are retained (under 1 MiB for one checkpoint-vocabulary token),
        so greedy argmax and first-prefill-logit comparisons use every vocabulary ID.
        """
        def __init__(self, model_dir, chunk_rows=16384, key='lm_head.weight'):
            super().__init__()
            if chunk_rows < 1:
                raise ValueError('head chunk_rows must be positive')
            self.path, shape = locate_tensor(model_dir, key)
            self.out_features, self.in_features = shape
            self.key = key
            self.chunk_rows = chunk_rows

        @torch.no_grad()
        def forward(self, hidden):
            if hidden.device.type != 'cpu' or hidden.dtype != torch.float32:
                raise ValueError('streamed reference head requires CPU FP32 hidden states')
            if hidden.shape[-1] != self.in_features:
                raise ValueError('hidden width does not match checkpoint LM head')
            result = torch.empty((*hidden.shape[:-1], self.out_features), dtype=torch.float32)
            with safe_open(self.path, framework='pt', device='cpu') as checkpoint:
                source = checkpoint.get_slice(self.key)
                for first in range(0, self.out_features, self.chunk_rows):
                    end = min(first+self.chunk_rows, self.out_features)
                    weight = source[first:end, :].float()
                    result[..., first:end] = torch.nn.functional.linear(hidden, weight)
                    del weight
            return result


    def bind_streamed_tables(model, model_dir, chunk_rows=16384):
        """Bind tables into an official MiniCPMV4_7ForConditionalGeneration wrapper."""
        embedding = StreamedEmbedding(model_dir)
        head = StreamedLMHead(model_dir, chunk_rows)
        cfg = model.config.text_config
        if (embedding.num_embeddings, embedding.embedding_dim) != (cfg.vocab_size, cfg.hidden_size):
            raise ValueError('embedding matrix differs from text config')
        if (head.out_features, head.in_features) != (cfg.vocab_size, cfg.hidden_size):
            raise ValueError('LM-head matrix differs from text config')
        model.model.language_model.embed_tokens = embedding
        model.lm_head = head


    def self_test():
        """Tiny BF16 checkpoint, official hybrid decoder, scatter, cache and greedy parity."""
        import sys
        import tempfile
        from safetensors.torch import save_file
        from transformers import MiniCPMV4_7Config
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7ForConditionalGeneration
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'export'))
        from minicpm_v47_export import LanguageOnly, load_language_only
        torch.set_num_threads(1)
        torch.manual_seed(47)
        cfg = MiniCPMV4_7Config(
            image_token_id=20, image_start_id=21, image_end_id=22,
            slice_start_id=23, slice_end_id=24, newline_id=25,
            text_config={'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,
            'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':32,
            'vocab_size':129,'linear_num_key_heads':2,'linear_num_value_heads':2,
            'linear_key_head_dim':16,'linear_value_head_dim':16,'linear_conv_kernel_dim':4,
            'layer_types':['linear_attention','full_attention'],'partial_rotary_factor':1.0,
            'rope_parameters':{'rope_type':'default','rope_theta':10000000.0,
                               'partial_rotary_factor':1.0,'mrope_section':[6,5,5]},
            'tie_word_embeddings':False})
        cfg._attn_implementation='eager'; cfg.text_config._attn_implementation='eager'
        initial = LanguageOnly(cfg).eval()
        with tempfile.TemporaryDirectory() as td:
            state = {k:v.to(torch.bfloat16).contiguous() for k,v in initial.state_dict().items()}
            save_file(state, str(Path(td)/'model.safetensors'))
            del initial,state
            wrappers=[]
            for streaming in (False,True):
                with torch.device('meta'):
                    model=MiniCPMV4_7ForConditionalGeneration(cfg)
                loaded=load_language_only(td,cfg,component='decoder' if streaming else 'all')
                model.model.language_model=loaded.model.language_model
                model.lm_head=loaded.lm_head
                del loaded
                if streaming: bind_streamed_tables(model,td,chunk_rows=17)
                wrappers.append(model.eval())
            dense,streamed=wrappers
            ids=torch.tensor([[0,5,5,128,20,0]],dtype=torch.long)
            torch.testing.assert_close(dense.model.language_model.embed_tokens(ids),
                                       streamed.model.language_model.embed_tokens(ids),rtol=0,atol=0)
            for shape in ((1,1,64),(2,3,64)):
                hidden=torch.randn(shape)
                torch.testing.assert_close(streamed.lm_head(hidden),dense.lm_head(hidden),atol=2e-6,rtol=2e-5)
            prompt=torch.tensor([[3,21,20,20,20,20,22,4]])
            mm=(prompt==20).int()
            visual=torch.randn(4,64)
            outputs=[]
            with torch.no_grad():
                for model in wrappers:
                    outputs.append(model(input_ids=prompt,mm_token_type_ids=mm,target_sizes=torch.tensor([[8,8]]),
                         mm_encoder_outputs={'image':BaseModelOutputWithPooling(pooler_output=(visual,))},
                         use_cache=True,logits_to_keep=1))
                max_error=0.0
                for step in range(6):
                    a,b=outputs
                    max_error=max(max_error,float((a.logits-b.logits).abs().max()))
                    torch.testing.assert_close(a.logits,b.logits,atol=3e-6,rtol=2e-5)
                    token_a=a.logits[:,-1].argmax(-1,keepdim=True)
                    token_b=b.logits[:,-1].argmax(-1,keepdim=True)
                    torch.testing.assert_close(token_a,token_b,rtol=0,atol=0)
                    if step<5:
                        outputs=[m(input_ids=token_a,past_key_values=o.past_key_values,use_cache=True,logits_to_keep=1)
                                 for m,o in zip(wrappers,outputs)]
            assert not list(streamed.model.language_model.embed_tokens.parameters())
            assert not list(streamed.lm_head.parameters())
            print(f'PASS: tiny BF16 streamed embedding/head and official hybrid VQA prefill +5 cached decode steps; max logit abs error={max_error:.9g}')
            print('Synthetic parity only; no real model weights loaded.')
    return types.SimpleNamespace(**locals())
# Shared native build lookup. Missing probes are always an error.
def probe_path(name, kind):
    build = Path(os.environ.get('NCNN_LLM_BUILD_DIR',
                         str(Path(__file__).resolve().parents[1] / 'build')))
    if kind == 'library':
        filename = (name + '.dll' if sys.platform == 'win32' else
                    'lib' + name + ('.dylib' if sys.platform == 'darwin' else '.so'))
    else:
        filename = name + ('.exe' if sys.platform == 'win32' else '')
    for directory in (build, build / 'Release', build / 'Debug'):
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Build the CMake target {name} and set NCNN_LLM_BUILD_DIR; '
                            f'{filename} was not found in {build}')



def _v47_text_reference():
    #!/usr/bin/env python3
    """Official CPU VQA reference using separately verified HF visual features.

    This is a test oracle, never a runtime backend. Splitting the official model at
    its public mm_encoder_outputs interface avoids holding both FP32 towers in RAM.
    The feature file must come from the official vision reference for this image.
    """
    import argparse
    import json
    from pathlib import Path
    import sys
    import time

    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoConfig, AutoProcessor, GenerationConfig
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7ForConditionalGeneration
    from transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6 import MiniCPMV4_6ImageProcessorPil

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
    from minicpm_v47_export import load_language_only
    bind_streamed_tables = _streamed_reference().bind_streamed_tables


    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("model")
        parser.add_argument("image")
        parser.add_argument("hf_features")
        parser.add_argument("output")
        parser.add_argument("--tokens", type=int, default=32)
        parser.add_argument("--threads", type=int, default=2)
        parser.add_argument("--memory-mode", choices=("streamed", "full"), default="streamed",
                            help="streamed retains only decoder weights, reads embedding rows/head chunks on demand")
        parser.add_argument("--head-chunk-rows", type=int, default=16384)
        parser.add_argument("--prompt", default="Describe this image in one short sentence.")
        args = parser.parse_args()
        if args.tokens < 1 or args.threads < 1 or args.head_chunk_rows < 1:
            parser.error("--tokens, --threads and --head-chunk-rows must be positive")
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        processor.image_processor = MiniCPMV4_6ImageProcessorPil.from_pretrained(args.model, local_files_only=True)
        # The checkpoint's PAD is also an EOS token. Do not infer attention masks
        # by comparing token IDs to PAD, or silently substitute a different token.
        # Single-image oracle deliberately uses no padding; decode below relies on it.
        image = Image.open(args.image).convert("RGB")
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "image", "image": image},
                                                    {"type": "text", "text": args.prompt}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", downsample_mode="16x", max_slice_nums=1,
            use_image_id=False, enable_thinking=False, padding=False)
        cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
        required = ("input_ids", "pixel_values", "target_sizes", "mm_token_type_ids")
        if any(name not in inputs for name in required):
            raise ValueError(f"processor is missing required fields: {set(required) - set(inputs)}")
        if inputs["input_ids"].shape[0] != 1 or inputs["target_sizes"].shape != (1, 2):
            raise ValueError("reference supports exactly one unpadded image thumbnail")
        if "attention_mask" in inputs and not bool(inputs["attention_mask"].all()):
            raise ValueError("padded reference prompts are unsupported")
        mask = inputs["input_ids"] == cfg.image_token_id
        if not torch.equal(inputs["mm_token_type_ids"] == 1, mask):
            raise ValueError("processor image modality labels disagree with image tokens")
        grid = inputs["target_sizes"][0]
        if bool((grid % 4 != 0).any()) or int(mask.sum()) != int(grid.prod()) // 16:
            raise ValueError("processor visual token count does not match 16x target grid")
        for marker in (cfg.image_start_id, cfg.image_end_id):
            if int((inputs["input_ids"] == marker).sum()) != 1:
                raise ValueError("expected one image start and end marker")
        if any(bool((inputs["input_ids"] == marker).any()) for marker in
               (cfg.slice_start_id, cfg.slice_end_id)):
            raise ValueError("detail slices are unsupported in this reference")
        (output / "prompt.txt").write_text(processor.decode(inputs["input_ids"][0],
                                                           skip_special_tokens=False))
        for name in ("input_ids", "pixel_values", "target_sizes", "mm_token_type_ids"):
            if name in inputs:
                np.save(output / (name + ".npy"), inputs[name].numpy())
        features = torch.from_numpy(np.load(args.hf_features, allow_pickle=False)).float()
        if features.ndim == 3 and features.shape[0] == 1:
            features = features[0]
        expected_shape = (int(mask.sum()), cfg.text_config.hidden_size)
        if tuple(features.shape) != expected_shape or not bool(torch.isfinite(features).all()):
            raise ValueError(f"invalid official visual features: {tuple(features.shape)}, expected {expected_shape}")
        # from_config/meta construction does NOT load generation_config.json.
        # Preserve both checkpoint stop IDs, not merely tokenizer.eos_token_id.
        generation_config = GenerationConfig.from_pretrained(args.model, local_files_only=True)
        eos_value = generation_config.eos_token_id
        if eos_value is None:
            eos_value = processor.tokenizer.eos_token_id
        eos = {eos_value} if isinstance(eos_value, int) else set(eos_value or [])
        if not eos:
            raise ValueError("no checkpoint EOS IDs available")
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = processor.tokenizer.pad_token_id
        print(f"Prepared {inputs['input_ids'].shape[-1]} prompt tokens, "
              f"{features.shape[0]} official visual tokens; EOS={sorted(eos)}", flush=True)
        cfg._attn_implementation = "eager"
        cfg.text_config._attn_implementation = "eager"
        with torch.device("meta"):
            model = MiniCPMV4_7ForConditionalGeneration(cfg)
        language = load_language_only(args.model, cfg,
                                      component="decoder" if args.memory_mode == "streamed" else "all")
        model.model.language_model = language.model.language_model
        model.lm_head = language.lm_head
        del language
        if args.memory_mode == "streamed":
            bind_streamed_tables(model, args.model, args.head_chunk_rows)
        model.generation_config = generation_config
        model.eval()
        assert not any(p.is_meta for p in model.model.language_model.parameters())
        assert all(p.is_meta for p in model.model.vision_tower.parameters())
        assert all(p.is_meta for p in model.model.merger.parameters())
        assert not any(p.is_meta for p in model.lm_head.parameters())
        # Fail visibly if supplied features accidentally stop bypassing either tower.
        def reject_vision_call(module, args):
            raise RuntimeError("precomputed visual reference must not execute meta vision modules")
        vision_hooks = [model.model.vision_tower.register_forward_pre_hook(reject_vision_call),
                        model.model.merger.register_forward_pre_hook(reject_vision_call)]
        print("Loaded real language weights; using official precomputed visual features", flush=True)
        inputs.pop("pixel_values", None)
        inputs["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=(features,))}
        inputs["downsample_mode"] = "16x"
        positions, deltas = model.model.get_rope_index(
            inputs["input_ids"], mm_token_type_ids=inputs.get("mm_token_type_ids"),
            target_sizes=inputs["target_sizes"], downsample_mode="16x",
            attention_mask=inputs.get("attention_mask"))
        np.save(output / "positions.npy", positions.numpy())
        generated = []
        started = time.time()
        with torch.no_grad():
            result = model(**inputs, use_cache=True, logits_to_keep=1)
            if result.past_key_values is None:
                raise RuntimeError("official prefill returned no cache")
            if result.past_key_values.get_seq_length() != inputs["input_ids"].shape[-1]:
                raise RuntimeError("unexpected prefill cache sequence length")
            torch.testing.assert_close(model.model.rope_deltas, deltas, rtol=0, atol=0)
            if not bool(torch.isfinite(result.logits).all()):
                raise RuntimeError("non-finite prefill logits")
            np.save(output / "prefill_logits.npy", result.logits[0, -1].float().numpy())
            for step in range(args.tokens):
                token = int(result.logits[0, -1].argmax())
                generated.append(token)
                print("token", step, token, repr(processor.decode([token])), flush=True)
                if token in eos or step + 1 == args.tokens:
                    break
                previous_length = result.past_key_values.get_seq_length()
                # No attention_mask on decode is intentional: this oracle rejects
                # padding. The official model derives one new canvas position from
                # cache length + its saved rope_delta. Passing a full-length mask
                # directly would produce full-length position IDs in this HF method.
                result = model(input_ids=torch.tensor([[token]], dtype=torch.long),
                               past_key_values=result.past_key_values,
                               use_cache=True, logits_to_keep=1)
                if result.past_key_values.get_seq_length() != previous_length + 1:
                    raise RuntimeError("official decode did not advance cache by one")
                if not bool(torch.isfinite(result.logits).all()):
                    raise RuntimeError("non-finite decode logits")
        for hook in vision_hooks:
            hook.remove()
        report = {"generated_ids": generated,
                  "text": processor.decode(generated, skip_special_tokens=True),
                  "seconds": time.time() - started, "image": str(args.image),
                  "input_ids": inputs["input_ids"][0].tolist(),
                  "image_processor": type(processor.image_processor).__name__,
                  "reference": "official HF forward with official precomputed vision features",
                  "dtype": "float32", "device": "cpu", "rope_deltas": deltas.tolist(),
                  "eos_token_ids": sorted(eos), "pad_token_id": generation_config.pad_token_id,
                  "stop_reason": "eos" if generated[-1] in eos else "max_tokens",
                  "hf_features": str(args.hf_features), "visual_feature_shape": list(features.shape),
                  "memory_mode": args.memory_mode, "head_chunk_rows": args.head_chunk_rows,
                  "streamed_operations": ["checkpoint embedding rows", "FP32 chunked linear head"]
                     if args.memory_mode == "streamed" else []}
        (output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(report["text"], flush=True)
    return types.SimpleNamespace(**locals())


def _prepare_minicpm_decoder_probe():
    """Prepare/compare a saved synthetic tiny decoder, without weights/network.

    Usage: python tests/test_minicpm.py prepare-decoder FIXTURE_DIR CASE_DIR [PROBE]
    The fixture is check_trace.py's tiny 64-wide, four-layer decoder.ts/param/bin.
    Torch wrapper references share export logic; the separate HF model comparison
    is recorded explicitly, not silently substituted for a failed reference.
    """
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import numpy as np
    import torch
    from transformers import MiniCPMV4_7Config
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'export'))
    from minicpm_v47_export import LanguageOnly
    from qwen35_export import Qwen35Decoder, _rename_cache_io
    from add_kvcache import add_kvcache_to_param


    def main():
        source=Path(sys.argv[1]); root=Path(sys.argv[2]); root.mkdir(parents=True,exist_ok=True)
        cfg=MiniCPMV4_7Config(text_config={'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,'num_hidden_layers':4,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':32,'vocab_size':256,'linear_num_key_heads':2,'linear_num_value_heads':2,'linear_key_head_dim':16,'linear_value_head_dim':16,'linear_conv_kernel_dim':4,'layer_types':['linear_attention']*3+['full_attention'],'partial_rotary_factor':1.0,'rope_parameters':{'rope_type':'default','rope_theta':10000000.0,'partial_rotary_factor':1.0,'mrope_section':[6,5,5]},'tie_word_embeddings':False})
        torch.set_num_threads(2); torch.manual_seed(913)
        model=LanguageOnly(cfg).eval().requires_grad_(False); wrapper=Qwen35Decoder(model).eval()
        traced=torch.jit.load(str(source/'decoder.ts')).eval()
        state=traced.state_dict()
        # TorchScript persists rotary buffers that HF registers non-persistent.
        for name in ('inv_freq','original_inv_freq'):
            key='language_model.rotary_emb.'+name
            if key in state:
                torch.testing.assert_close(state.pop(key),getattr(wrapper.language_model.rotary_emb,name),rtol=0,atol=0)
        wrapper.load_state_dict(state,strict=True)
        print('Loaded exact synthetic checkpoint state',flush=True)
        shutil.copyfile(source/'decoder.ncnn.param',root/'decoder.ncnn.param')
        shutil.copyfile(source/'decoder.ncnn.bin',root/'decoder.ncnn.bin')
        _rename_cache_io(root/'decoder.ncnn.param',3,3)
        add_kvcache_to_param(str(root/'decoder.ncnn.param'),scale=None,backup=False)
        hidden=torch.randn(1,12,64)*.25
        pos=torch.arange(12)[None,None].expand(3,1,-1)
        cos,sin=model.model.language_model.rotary_emb(hidden,pos)
        zero=[torch.zeros(1,96,4) for _ in range(3)]+[torch.zeros(1,2,16,16) for _ in range(3)]
        initial={**{f'cache_conv{i}':{'shape':[4,96],'zero':True} for i in range(3)},
                 **{f'cache_gdr{i}':{'shape':[16,16,2],'zero':True} for i in range(3)},
                 'cache_k0':None,'cache_v0':None}
        outputs={key:'out_'+key for key in initial}
        case={'initial_states':initial,'state_outputs':outputs,'steps':[]}
        references={}; report={'synthetic':True,'trained_model':False,'hf_reference':[]}
        def save(name,value):
            array=value.detach().cpu().numpy().astype('<f4'); array.tofile(root/(name+'.f32'))
            return {'file':name+'.f32','shape':list(reversed(array.shape))}
        start=0
        for index,end in enumerate((8,9,12)):
            causal=torch.triu(torch.full((end,end),float('-inf')),diagonal=1)
            args=(hidden[:,:end],causal[None,None],cos[:,:end,:16],sin[:,:end,:16],*zero)
            with torch.no_grad():
                ref=wrapper(*args)
                hf=model.model.language_model(inputs_embeds=hidden[:,:end],position_ids=pos[:,:,:end],use_cache=False).last_hidden_state
            np.testing.assert_allclose(ref[0].numpy(),hf.numpy(),atol=2e-5,rtol=2e-5)
            report['hf_reference'].append({'length':end,'max_abs':float((ref[0]-hf).abs().max())})
            if index==0:
                with torch.no_grad(): ts=traced(*args)
                for left,right in zip(ref,ts): np.testing.assert_allclose(left.numpy(),right.numpy(),atol=1e-6,rtol=1e-6)
            inputs={'in0':save(f'{index}_hidden',hidden[0,start:end]),
                    'in1':save(f'{index}_mask',causal[start:end]),
                    'in2':save(f'{index}_cos',cos[0,start:end,:16]),
                    'in3':save(f'{index}_sin',sin[0,start:end,:16])}
            case['steps'].append({'inputs':inputs})
            refs={'out0':save(f'ref_{index}_out0',ref[0][0,start:end])}
            for i in range(3):
                refs[f'out_cache_conv{i}']=save(f'ref_{index}_conv{i}',ref[1+i][0])
                refs[f'out_cache_gdr{i}']=save(f'ref_{index}_gdr{i}',ref[4+i][0])
            references[str(index)]=refs; start=end
        (root/'case.json').write_text(json.dumps(case,indent=2))
        (root/'references.json').write_text(json.dumps(references,indent=2))
        if len(sys.argv)>3:
            destination=root/'native'
            subprocess.run([sys.argv[3],str(root/'decoder.ncnn.param'),str(root/'decoder.ncnn.bin'),str(root/'case.json'),str(destination)],check=True)
            report['native_parity']=compare_cached_decoder(root)['native_parity']
        (root/'parity-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    return types.SimpleNamespace(**locals())


def _preprocess_reference():
    """Measure complete C++ preprocessing against actual HF Torchvision/PIL backends."""
    import argparse
    import ctypes
    import io
    import json
    import pathlib
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor
    from transformers.models.minicpmv4_6.image_processing_minicpmv4_6 import MiniCPMV4_6ImageProcessor
    from transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6 import MiniCPMV4_6ImageProcessorPil


    def main():
        torch.set_num_threads(1)
        ap=argparse.ArgumentParser()
        ap.add_argument('--model-dir',required=True)
        ap.add_argument('--output')
        ap.add_argument('--fixture', help='Optional PNG to save actual processor tensors for')
        ap.add_argument('--fixture-output-prefix', default='processor')
        args=ap.parse_args()
        processor=AutoImageProcessor.from_pretrained(args.model_dir,local_files_only=True)
        print('Auto processor:',type(processor).__name__,flush=True)
        pil=MiniCPMV4_6ImageProcessorPil.from_pretrained(args.model_dir,local_files_only=True)
        processors={'auto':processor,'pil':pil}
        if not isinstance(processor,MiniCPMV4_6ImageProcessor):
            processors['torchvision']=MiniCPMV4_6ImageProcessor.from_pretrained(args.model_dir,local_files_only=True)
        if args.fixture:
            image=Image.open(args.fixture).convert('RGB')
            for name,p in processors.items():
                out=p(images=image,max_slice_nums=1,use_image_id=False,return_tensors='pt')
                prefix=args.fixture_output_prefix+('-pil' if name=='pil' else '')
                np.save(prefix+'-pixels.npy',out['pixel_values'].float().cpu().numpy())
                np.save(prefix+'-target-sizes.npy',out['target_sizes'].cpu().numpy())
                print('Saved actual processor tensors:',prefix,flush=True)
        rng=np.random.default_rng(4710)
        dims=[(448,448),(100,100),(300,400),(400,300),(900,1200),(1200,900),(53,511),(511,53),(1,1),(56,56)]
        results=[]
        so = probe_path('minicpm_preprocess_probe', 'library')
        run=ctypes.CDLL(str(so)).minicpm_preprocess_probe
        up=ctypes.POINTER(ctypes.c_ubyte); fp=ctypes.POINTER(ctypes.c_float); ip=ctypes.POINTER(ctypes.c_int)
        run.argtypes=[up,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,fp,ctypes.c_int,ip]
        for h,w in dims:
            for pattern in ('random','checkerboard'):
                rgb=rng.integers(0,256,(h,w,3),dtype=np.uint8)
                if pattern=='checkerboard': rgb=np.repeat(((np.indices((h,w)).sum(0)%2)*255).astype(np.uint8)[:,:,None],3,axis=2)
                # Exercise real lossless RGB image decoding, not an already normalized tensor.
                buf=io.BytesIO(); Image.fromarray(rgb).save(buf,format='PNG'); buf.seek(0)
                image=Image.open(buf).convert('RGB')
                actual=None
                for name,p in processors.items():
                    out=p(images=image,max_slice_nums=1,use_image_id=False,return_tensors='pt')
                    ref=out['pixel_values'].float().cpu().numpy().reshape(-1)
                    if actual is None:
                        actual=np.empty(ref.size,dtype=np.float32); shape=np.zeros(2,dtype=np.int32)
                        size=run(rgb.ctypes.data_as(up),h,w,p.patch_size,p.scale_resolution,
                                 actual.ctypes.data_as(fp),actual.size,shape.ctypes.data_as(ip))
                        assert size==ref.size,(size,ref.size)
                    np.testing.assert_array_equal(shape,out['target_sizes'][0].cpu().numpy())
                    diff=np.abs(actual-ref)
                    # Quantify resizing separately from float normalization rounding.
                    actual_u8=np.rint((actual+1)*127.5).astype(np.uint8)
                    ref_u8=np.rint((ref+1)*127.5).astype(np.uint8)
                    item={'backend':name,'class':type(p).__name__,'height':h,'width':w,'pattern':pattern,
                          'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
                          'different_pixels':int(np.count_nonzero(actual_u8!=ref_u8)),
                          'max_byte_difference':int(np.max(np.abs(actual_u8.astype(int)-ref_u8.astype(int))))}
                    results.append(item)
                    print(json.dumps(item),flush=True)
        summary={name:{'max_abs':max(r['max_abs'] for r in results if r['backend']==name),
                       'max_byte_difference':max(r['max_byte_difference'] for r in results if r['backend']==name),
                       'different_pixels':sum(r['different_pixels'] for r in results if r['backend']==name)}
                 for name in processors}
        print('SUMMARY',json.dumps(summary),flush=True)
        if args.output: pathlib.Path(args.output).write_text(json.dumps({'results':results,'summary':summary},indent=2))
        # PIL backend is the explicitly selected reference for this helper. Report
        # default Torchvision discrepancies rather than silently treating them as parity.
        assert summary['pil']['different_pixels']==0,summary
        assert summary['pil']['max_abs']<3e-7,summary
    return types.SimpleNamespace(**locals())


def _prompt():
    """Compare the C++ MiniCPM prompt with the checkpoint's Jinja and token IDs.
    No torch/transformers imports or weights. Usage: python TEST MODEL_DIR PROBE
    """
    import itertools
    import json
    import subprocess
    import sys
    from pathlib import Path
    from jinja2 import Environment, StrictUndefined
    from tokenizers import Tokenizer

    model, probe = Path(sys.argv[1]), sys.argv[2]
    def fail(message):
        raise ValueError(message)
    env = Environment()
    env.globals['raise_exception'] = fail
    template = env.from_string(json.loads((model/'tokenizer_config.json').read_text())['chat_template'])
    tokenizer = Tokenizer.from_file(str(model/'tokenizer.json'))
    cases = []
    for thinking, generation, system, history in itertools.product((False,True),repeat=4):
        messages = ([{'role':'system','content':'  You are a helpful assistant.\n'}] if system else [])
        if history:
            messages += [{'role':'user','content':' First question? '},
                         {'role':'assistant','content':' First answer. ','reasoning_content':' secret past thought '}]
        messages += [{'role':'user','content':' <image>\nDescribe this image in one short sentence.\n'}]
        cases.append(dict(messages=messages,enable_thinking=thinking,add_generation_prompt=generation))
    for content, reasoning in [('Answer.', ' final thought '), ('<think>\nthought\n</think>\n\nAnswer.', None), ('Answer.',None)]:
        for thinking in (False,True):
            assistant = {'role':'assistant','content':content}
            if reasoning is not None: assistant['reasoning_content']=reasoning
            cases.append(dict(messages=[{'role':'user','content':'Question'},assistant], enable_thinking=thinking,add_generation_prompt=False))
    for content in ['a  b', 'a\n\nb', '中文问题：图片里有什么？', '1234567890', ' a\t b ']:
        cases.append(dict(messages=[{'role':'user','content':content}],enable_thinking=False,add_generation_prompt=True))
    process=subprocess.run([probe],input=''.join(json.dumps(c)+'\n' for c in cases),text=True,capture_output=True,check=True)
    results=[json.loads(line) for line in process.stdout.splitlines()]
    assert len(results)==len(cases)
    for i,(case,actual) in enumerate(zip(cases,results)):
        expected=template.render(**case)
        assert actual.get('text')==expected,(i,actual,repr(expected))
        a=tokenizer.encode(actual['text'],add_special_tokens=False).ids
        b=tokenizer.encode(expected,add_special_tokens=False).ids
        assert a==b,(i,a,b)
        # The checkpoint's ByteLevel postprocessor adds no BOS, despite bos_token metadata.
        assert tokenizer.encode(expected,add_special_tokens=True).ids==b
    print(json.dumps({'passed':len(cases),'exact_strings':True,'official_tokenizer_ids_for_rendered_text':True,'automatic_bos':False}))
    return types.SimpleNamespace(**locals())


def _resize_reference():
    """Byte-exact comparison of C++ thumbnail bicubic resize against Pillow.
    No model weights or Torch needed. Uses the CMake-built image utility probe.
    """
    import ctypes
    import numpy as np
    import PIL
    from PIL import Image


    def main():
        rng=np.random.default_rng(470123)
        dimensions=[(1,1,3,5),(1,7,9,3),(7,1,3,9),(2,2,1,1),(3,7,3,7),
                    (200,300,448,672),(900,1200,392,504),(1200,900,504,392),
                    (1000,1000,448,448),(56,56,448,448),(448,448,56,56),
                    (60,400,700,28),(400,60,28,700),(32,40,32,10),(32,40,8,40)]
        dimensions += [tuple(int(x) for x in rng.integers(1,200,size=4)) for _ in range(200)]
        so = probe_path('minicpm_resize_probe', 'library')
        lib=ctypes.CDLL(str(so))
        pointer=ctypes.POINTER(ctypes.c_ubyte)
        run=lib.minicpm_resize_probe
        run.argtypes=[pointer,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,pointer]
        run.restype=ctypes.c_int
        count=0
        for h,w,th,tw in dimensions:
            images=[rng.integers(0,256,size=(h,w,3),dtype=np.uint8)]
            # Sharp checkerboard and impulse exercise negative lobes/clipping.
            checker=((np.indices((h,w)).sum(axis=0)%2)*255).astype(np.uint8)
            images.append(np.repeat(checker[:,:,None],3,axis=2))
            impulse=np.zeros((h,w,3),dtype=np.uint8); impulse[h//2,w//2]=[255,127,61]
            images.append(impulse)
            for original in images:
                expected=np.asarray(Image.fromarray(original).resize((tw,th),Image.Resampling.BICUBIC))
                for bgr in (False,True):
                    src=np.ascontiguousarray(original[:,:,::-1] if bgr else original)
                    actual=np.empty((th,tw,3),dtype=np.uint8)
                    status=run(src.ctypes.data_as(pointer),h,w,th,tw,bgr,actual.ctypes.data_as(pointer))
                    assert status==0
                    np.testing.assert_array_equal(actual,expected,err_msg=str((h,w,th,tw,bgr)))
                    count+=1
        dummy=np.zeros(3,dtype=np.uint8)
        assert run(dummy.ctypes.data_as(pointer),0,1,1,1,0,dummy.ctypes.data_as(pointer))==-1
        assert run(dummy.ctypes.data_as(pointer),1,1,0,1,0,dummy.ctypes.data_as(pointer))==-1
        print(f'PASS: {count} byte-exact RGB/BGR bicubic comparisons against Pillow {PIL.__version__}; invalid dimensions rejected')
    return types.SimpleNamespace(**locals())


def _tokenizer_reference():
    """Actual native BPE ID parity with checkpoint tokenizers backend, no Torch."""
    import argparse
    import copy
    import json
    import os
    from pathlib import Path
    import random
    import subprocess
    import sys
    import tempfile
    from tokenizers import Tokenizer

    ROOT=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(ROOT/'export'))
    from minicpm_v47_tokenizer import export_minicpm_v47_tokenizer,validate_pipeline


    def cases(data):
        texts=['#include <stdio.h>\n#define FOO 42\n### Header\n########',r'\r\n\t',r'hello\rworld',r'\r',r'\n',r'\t','','a  b','a   b',' a b ','a\t\tb','a\r\n  b',' \n\t \r\n  ','\n\r\n',
               "I'm I'M i'd I'd I've I’M 'ſ 'S 't 're 've 'm 'll 'd",'1234567890 １２３ ①Ⅱ²',
               '你好，世界！中文 English123混合','🙂hello 👨‍👩‍👧‍👦!  🚀🚀abc','e\u0301 A\u030a \u212b',
               '각각한글','\u0315\u0300A\u0301\u0327','किताब हिन्दी বাংলা தமிழ்',
               'שלום עולם العربية فارسی','\x00A\x01\x1cB\x85C\u00a0D\u2003E\u2028F\u2029G\u3000H',
               'def foo(x=123):\n    return "a  b"\n\n# 中文注释\n','!!!abc !abc .\r\n ',
               '<|im_start|>user\n<image><|image_pad|></image>\nDescribe.<|im_end|>\n',
               '<think>\n\n</think>\n\n<tool_call>{"a":1}</tool_call>',
               '\U00011938\U0001e4ec\U0001e4ed\u0300']
        for item in data['added_tokens']:
            texts.extend([item['content'],'a '+item['content']+' b','e\u0301'+item['content']+'A\u030a'])
        rng=random.Random(47)
        whitespace=' \t\n\r\f\v\x85\u00a0\u1680\u2000\u200b\u2028\u2029\u202f\u205f\u3000'
        alphabet=list("abcXYZstrEVmLD'!?.,{}[]()/\\0123456789")+list(whitespace)+list('你好世界中文éÅßſİı각🙂🚀')
        alphabet+=list('\u0300\u0301\u0308\u0315\u0327\u0345\u093c\u05b0\u05b1')
        for _ in range(3000): texts.append(''.join(rng.choices(alphabet,k=rng.randrange(0,120))))
        # Random scalars cover classification and byte-level fallback independently
        # of ordinary language fixtures. Exclude surrogate code points only.
        for _ in range(500):
            row=[]
            for _ in range(rng.randrange(1,50)):
                cp=rng.randrange(0x110000)
                if not 0xd800<=cp<=0xdfff: row.append(chr(cp))
            texts.append(''.join(row))
        # Every non-Hangul decomposition and every class-table scalar is sampled
        # in surrounding text, exercising C++ NFC rather than only Python tables.
        return texts


    def main():
        ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True)
        ap.add_argument('--artifact-dir');ap.add_argument('--report')
        args=ap.parse_args()
        raw=Path(args.model_dir)/'tokenizer.json';data=json.loads(raw.read_text());validate_pipeline(data)
        for field, value in [('decoder', None), ('post_processor', None),
                             ('truncation', {}), ('padding', {})]:
            unsupported=copy.deepcopy(data);unsupported[field]=value
            try: validate_pipeline(unsupported)
            except ValueError: pass
            else: raise AssertionError('accepted unsupported '+field)
        reference=Tokenizer.from_file(str(raw))
        texts=cases(data)
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); artifact_dir=Path(args.artifact_dir) if args.artifact_dir else td
            artifact=artifact_dir/'minicpm_v47_tokenizer.txt'
            if not artifact.exists(): export_minicpm_v47_tokenizer(args.model_dir,artifact_dir)
            for line in artifact.read_text().splitlines():
                fields=line.split()
                if fields and fields[0] in ('D','C'):
                    cp=int(fields[1]);texts.append('a'+chr(cp)+'\u0301\u0327 b')
            by_id={v:k for k,v in data['model']['vocab'].items()}
            by_id.update({v['id']:v['content'] for v in data['added_tokens']})
            assert sorted(by_id)==list(range(len(by_id)))
            (td/'vocab.txt').write_text(''.join(by_id[i].replace('\n','\\n').replace('\r','\\r')+'\n' for i in range(len(by_id))))
            (td/'merges.txt').write_text(''.join((' '.join(m) if isinstance(m,list) else m)+'\n' for m in data['model']['merges']))
            probe=probe_path('minicpm_tokenizer_probe', 'executable')
            payload='\n'.join(t.encode().hex() for t in texts)+'\n'
            process=subprocess.run([str(probe),str(td/'vocab.txt'),str(td/'merges.txt'),str(artifact)],
                                   input=payload,text=True,capture_output=True,check=True)
            rows=process.stdout.splitlines();assert len(rows)==len(texts),(len(rows),len(texts),process.stderr)
            failures=[]
            for i,(text,row) in enumerate(zip(texts,rows)):
                if row.startswith('ERROR:'): failures.append({'index':i,'text':text,'error':row});continue
                raw_ids,decoded,skipped=row.split('|')
                actual=[int(x) for x in raw_ids.split(',')] if raw_ids else []
                expected=reference.encode(text,add_special_tokens=False).ids
                if actual!=expected:
                    failures.append({'index':i,'text':text,'actual':actual,'expected':expected})
                # decode byte sequences may be incomplete only for synthetic ID lists;
                # complete encoded inputs here should match backend normalization.
                if bytes.fromhex(decoded).decode('utf-8',errors='replace')!=reference.decode(expected,skip_special_tokens=False):
                    failures.append({'index':i,'text':text,'decode_mismatch':True})
                if bytes.fromhex(skipped).decode('utf-8',errors='replace')!=reference.decode(expected,skip_special_tokens=True):
                    failures.append({'index':i,'text':text,'skip_special_mismatch':True})
            # Fail closed for invalid UTF-8 and unsupported regex artifacts.
            bad=subprocess.run([str(probe),str(td/'vocab.txt'),str(td/'merges.txt'),str(artifact)],input='80\nc080\neda080\nf4908080\ne282\n',text=True,capture_output=True,check=True)
            assert len(bad.stdout.splitlines())==5 and all(line.startswith('ERROR:') for line in bad.stdout.splitlines())
            original=artifact.read_text()
            records=original.splitlines()
            mutations={'regex':original.replace("(?i:'s", "(?i:'x",1),
                       'missing_end':original.replace('END\n',''),
                       'source_hex':original.replace(next(r for r in records if r.startswith('source ')), 'source '+'z'*64),
                       'decomposition_count':original.replace(next(r for r in records if r.startswith('D ')), 'D 65 4294967295'),
                       'added_vocab_mismatch':original.replace(next(r for r in records if r.startswith('A ')), 'A 0 1 616263'),
                       'excess_fields':original.replace('backend ', 'backend extra ',1)}
            for kind in ('R','D','C','P','A'):
                record=next(r for r in records if r.startswith(kind+' '))
                mutations['duplicate_'+kind]=original.replace(record,record+'\n'+record,1)
            for name, content in mutations.items():
                invalid=td/(name+'.txt');invalid.write_text(content)
                rejected=subprocess.run([str(probe),str(td/'vocab.txt'),str(td/'merges.txt'),str(invalid)],input='',text=True,capture_output=True)
                assert rejected.returncode!=0, 'accepted invalid artifact '+name
        report={'case_count':len(texts),'failure_count':len(failures),'failures':failures,
                'reference':'actual tokenizers.Tokenizer.from_file(checkpoint/tokenizer.json)',
                'invalid_utf8_rejected':True,'unsupported_regex_rejected':True,
                'invalid_artifact_cases':list(mutations),'unsupported_pipeline_cases':4}
        if args.report: Path(args.report).write_text(json.dumps(report,ensure_ascii=True,indent=2))
        print(json.dumps({**report,'failures':failures[:5]},ensure_ascii=True,indent=2),flush=True)
        assert not failures,'native tokenizer differs from checkpoint backend'
    return types.SimpleNamespace(**locals())


# Official openbmb/MiniCPM-V-4.7-1B config, revision ef3ab3bf0120a60e1a2d2b52d984a79d8cb7c630.
SUPPORTED_CONFIG = {'architectures': ['MiniCPMV4_7ForConditionalGeneration'],
 'dtype': 'bfloat16',
 'model_type': 'minicpmv4_7',
 'tie_word_embeddings': False,
 'transformers_version': '5.18.0',
 'use_cache': True,
 'text_config': {'attention_bias': False,
                 'attention_dropout': 0.0,
                 'attn_output_gate': True,
                 'full_attention_interval': 4,
                 'head_dim': 256,
                 'hidden_act': 'silu',
                 'hidden_size': 1024,
                 'initializer_range': 0.02,
                 'intermediate_size': 3584,
                 'layer_types': ['linear_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'full_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'full_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'full_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'full_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'full_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'linear_attention',
                                 'full_attention'],
                 'linear_conv_kernel_dim': 4,
                 'linear_key_head_dim': 128,
                 'linear_num_key_heads': 16,
                 'linear_num_value_heads': 16,
                 'linear_value_head_dim': 128,
                 'mamba_ssm_dtype': 'float32',
                 'max_position_embeddings': 262144,
                 'mlp_only_layers': [],
                 'mtp_num_hidden_layers': 1,
                 'mtp_use_dedicated_embeddings': False,
                 'num_attention_heads': 8,
                 'num_hidden_layers': 24,
                 'num_key_value_heads': 2,
                 'partial_rotary_factor': 0.25,
                 'rms_norm_eps': 1e-06,
                 'rope_parameters': {'mrope_interleaved': True,
                                     'mrope_section': [11, 11, 10],
                                     'partial_rotary_factor': 0.25,
                                     'rope_theta': 10000000,
                                     'rope_type': 'default'},
                 'vocab_size': 248094,
                 'model_type': 'qwen3_5_text',
                 'tie_word_embeddings': False},
 'insert_layer_id': 6,
 'downsample_mode': '16x',
 'mrope_mode': 'canvas',
 'vision_config': {'attention_dropout': 0.0,
                   'hidden_act': 'gelu_pytorch_tanh',
                   'hidden_size': 1152,
                   'image_size': 980,
                   'intermediate_size': 4304,
                   'layer_norm_eps': 1e-06,
                   'model_type': 'minicpmv4_7_vision',
                   'num_attention_heads': 16,
                   'num_channels': 3,
                   'num_hidden_layers': 27,
                   'patch_size': 14},
 'image_token_id': 248056,
 'video_token_id': 248057,
 'image_start_id': 248078,
 'image_end_id': 248079,
 'slice_start_id': 248088,
 'slice_end_id': 248089,
 'newline_id': 198,
 'version': 4.7}

def _v47_language_export():
    """Synthetic converter contract tests; these are NOT trained model inference."""
    import sys, tempfile, unittest, json
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    import numpy as np
    from safetensors.torch import save_file
    from transformers import MiniCPMV4_7Config
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'export'))
    from minicpm_v47_export import LanguageOnly, load_language_only, finalize_runtime_config
    from qwen35_export import Qwen35Decoder, PortableGatedNorm
    from qwen35_gdr import torch_chunk_gated_delta_rule
    from minicpm_v47_language_export import export_tables

    class LanguageExportTests(unittest.TestCase):
     def test_runtime_preflight_before_any_export_or_staging(self):
      import copy
      import minicpm_v47_export as converter
      converter.validate_runtime_config(copy.deepcopy(SUPPORTED_CONFIG))
      mutations = [
       (('vision_config','patch_size'), 16), (('vision_config','num_channels'), 1),
       (('vision_config','num_attention_heads'), 0), (('vision_config','num_attention_heads'), 7),
       (('vision_config','num_hidden_layers'), 1), (('vision_config','num_hidden_layers'), 65),
       (('vision_config','image_size'), 981), (('vision_config','image_size'), 1806),
       (('vision_config','hidden_size'), 2049), (('vision_config','hidden_size'), True),
       (('insert_layer_id',), 26), (('insert_layer_id',), -1),
       (('downsample_mode',), '4x'), (('mrope_mode',), 'default'),
       (('vision_config','window_kernel_size'), [2,3]), (('merge_kernel_size',), [4,4]),
       (('merger_times',), 2), (('tie_word_embeddings',), True),
       (('text_config','tie_word_embeddings'), True), (('text_config','model_type'), 'other'),
       (('text_config','hidden_size'), 8193), (('text_config','layer_types'), ['full_attention']*24),
       (('text_config','rope_parameters','mrope_section'), [11,11,11]),
       (('text_config','rope_parameters','partial_rotary_factor'), 0),
       (('text_config','rope_parameters','rope_theta'), float('inf')),
       (('text_config','rope_parameters','mrope_interleaved'), False),
       (('text_config','rope_parameters','rope_type'), 'dynamic'),
       (('image_token_id',), 198), (('image_token_id',), 248094),
      ]
      with tempfile.TemporaryDirectory() as td:
       root=Path(td); model=root/'model'; model.mkdir(); output=root/'converted'
       for path, value in mutations:
        cfg=copy.deepcopy(SUPPORTED_CONFIG); node=cfg
        for key in path[:-1]: node=node[key]
        node[path[-1]]=value
        (model/'config.json').write_text(json.dumps(cfg))
        with self.subTest(path=path, value=value), patch.object(sys,'argv',['export','-m',str(model),'-o',str(output)]), \
             patch('minicpm_v47_language_export.export_language_components') as language, \
             patch.object(converter,'export') as decoder, patch.object(converter,'export_minicpm_v47_vision') as vision:
         with self.assertRaisesRegex(ValueError, 'unsupported MiniCPM runtime config'): converter.main()
         language.assert_not_called(); decoder.assert_not_called(); vision.assert_not_called()
         self.assertFalse(output.exists())
         self.assertFalse(list(root.glob('.converted.staging-*')))

     def test_cli_publishes_only_after_finalizers(self):
      import minicpm_v47_export as converter
      with tempfile.TemporaryDirectory() as td:
       root=Path(td); model=root/'model';model.mkdir()
       (model/'config.json').write_text(json.dumps(SUPPORTED_CONFIG))
       for fail in (True,False):
        output=root/('failed' if fail else 'complete')
        def fake_export(source,destination,**kwargs):
         self.assertFalse(output.exists())
         self.assertEqual(kwargs['prefix'],output.name)
         (Path(destination)/'model.json').write_text('{}')
        def finalizer(source,destination):
         self.assertFalse(output.exists())
         if fail: raise RuntimeError('simulated tokenizer failure')
         (Path(destination)/'tokenizer-ready').write_text('ready')
        with patch.object(sys,'argv',['export','-m',str(model),'-o',str(output)]), \
             patch('minicpm_v47_language_export.export_language_components'), \
             patch.object(converter,'export',side_effect=fake_export), \
             patch.object(converter,'finalize_runtime_config'), \
             patch.object(converter,'finalize_tokenizer',side_effect=finalizer):
         if fail:
          with self.assertRaisesRegex(RuntimeError,'simulated'): converter.main()
          self.assertFalse(output.exists())
          self.assertEqual(len(list(root.glob('.failed.staging-*'))),1)
         else:
          converter.main()
          self.assertTrue((output/'tokenizer-ready').is_file())
          self.assertFalse(list(root.glob('.complete.staging-*')))
          with self.assertRaisesRegex(ValueError,'already exists'): converter.main()
     def test_tables_never_reuse_same_shape_stale_weights(self):
      with tempfile.TemporaryDirectory() as td:
       root=Path(td); output=root/'out'
       for value in (1.,2.):
        save_file({'model.language_model.embed_tokens.weight':torch.full((4,2),value),
                   'lm_head.weight':torch.full((4,2),value+1)},str(root/'model.safetensors'))
        export_tables(root,output,SimpleNamespace(vocab_size=4,hidden_size=2),'model')
        np.testing.assert_array_equal(np.fromfile(output/'model_embed_token.ncnn.bin',dtype='<f4',offset=4),np.full(8,value,dtype=np.float32))
        np.testing.assert_array_equal(np.fromfile(output/'model_proj_out.ncnn.bin',dtype='<f4',offset=4),np.full(8,value+1,dtype=np.float32))
     def test_runtime_has_no_implicit_bos(self):
      with tempfile.TemporaryDirectory() as td:
       root=Path(td); model=root/'model'; output=root/'output';model.mkdir();output.mkdir()
       config={'tokenizer':{'bos':'<|im_start|>','eos_ids':[7],'additional_special_tokens':['<|im_start|>']},'setting':{'functions':{'type':'tool_call'}}}
       (output/'model.json').write_text(json.dumps(config))
       finalize_runtime_config(model,output)
       result=json.loads((output/'model.json').read_text())
       self.assertEqual(result['tokenizer']['bos'],'')
       self.assertNotIn('functions',result['setting'])
       self.assertEqual(result['tokenizer']['additional_special_tokens'],['<|im_start|>'])
       self.assertEqual(result['tokenizer']['eos_ids'],[7])
       (model/'generation_config.json').write_text(json.dumps({'eos_token_id':[248044,248046]}))
       finalize_runtime_config(model,output)
       self.assertEqual(json.loads((output/'model.json').read_text())['tokenizer']['eos_ids'],[248044,248046])
     def test_stream_load_and_decoder_reference(self):
      torch.manual_seed(4); torch.set_num_threads(2)
      cfg=MiniCPMV4_7Config(text_config={'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,'num_hidden_layers':4,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':32,'vocab_size':256,'linear_num_key_heads':2,'linear_num_value_heads':2,'linear_key_head_dim':16,'linear_value_head_dim':16,'linear_conv_kernel_dim':4,'layer_types':['linear_attention']*3+['full_attention'],'partial_rotary_factor':1.0,'rope_parameters':{'rope_type':'default','rope_theta':10000000.0,'partial_rotary_factor':1.0,'mrope_section':[6,5,5]},'tie_word_embeddings':False})
      original=LanguageOnly(cfg).eval()
      with tempfile.TemporaryDirectory() as td:
       save_file(original.state_dict(),str(Path(td)/'model.safetensors'))
       loaded=load_language_only(td,cfg)
       for n,v in original.named_parameters(): torch.testing.assert_close(v,dict(loaded.named_parameters())[n],atol=0,rtol=0)
       decoder_only=load_language_only(td,cfg,component='decoder')
       self.assertEqual(decoder_only.model.language_model.embed_tokens.weight.shape[0],1)
       self.assertTrue(decoder_only.lm_head.weight.is_meta)
       for n,v in original.model.language_model.layers.named_parameters():
        torch.testing.assert_close(v,dict(decoder_only.model.language_model.layers.named_parameters())[n],atol=0,rtol=0)
       out=Path(td)/'ncnn';export_tables(td,out,cfg.text_config,'tiny')
       import ncnn
       for name,input_data,expected in (
        ('embed_token',np.array([0,7,255,2],dtype=np.int32),original.model.language_model.embed_tokens(torch.tensor([0,7,255,2])).detach().numpy()),
        ('proj_out',np.random.default_rng(2).normal(size=(3,64)).astype(np.float32),None)):
        if expected is None:expected=input_data@original.lm_head.weight.detach().numpy().T
        net=ncnn.Net();net.opt.num_threads=2;net.opt.use_fp16_storage=False;net.opt.use_fp16_packed=False;net.opt.use_bf16_storage=False;net.opt.use_packing_layout=False
        self.assertEqual(net.load_param(str(out/f'tiny_{name}.ncnn.param')),0)
        self.assertEqual(net.load_model(str(out/f'tiny_{name}.ncnn.bin')),0)
        ex=net.create_extractor();owner=np.ascontiguousarray(input_data);mat=ncnn.Mat(owner).clone()
        self.assertEqual(ex.input('in0',mat),0);ret,result=ex.extract('out0');self.assertEqual(ret,0)
        np.testing.assert_allclose(np.array(result).copy(),expected,atol=2e-5,rtol=2e-5)
      for layer in loaded.model.language_model.layers:
       if getattr(layer,'layer_type',getattr(layer,'block_type',None))=='linear_attention':
        layer.linear_attn.chunk_gated_delta_rule=torch_chunk_gated_delta_rule
        layer.linear_attn.recurrent_gated_delta_rule=torch_chunk_gated_delta_rule
        layer.linear_attn.norm=PortableGatedNorm(layer.linear_attn.norm)
      wrapper=Qwen35Decoder(loaded).eval(); lm=loaded.model.language_model
      ids=torch.arange(8)[None]; hidden=lm.embed_tokens(ids)
      pos=torch.arange(8)[None,None].expand(3,1,-1); cos,sin=lm.rotary_emb(hidden,pos)
      mask=torch.triu(torch.full((8,8),float('-inf')),diagonal=1)[None,None]
      states=[torch.zeros(1,96,4) for _ in range(3)]+[torch.zeros(1,2,16,16) for _ in range(3)]
      with torch.no_grad():
       expected=lm(inputs_embeds=hidden,use_cache=False).last_hidden_state
       actual=wrapper(hidden,mask,cos[...,:16],sin[...,:16],*states)[0]
      torch.testing.assert_close(actual,expected,atol=3e-3,rtol=3e-3)
      self.assertFalse(any(x.is_meta for x in loaded.parameters()))
      self.assertFalse(any(x.is_meta for x in loaded.buffers()))
    return types.SimpleNamespace(**locals())


def _v47_vision_export():
    """Synthetic graph/serialization tests. No model weights or network required.

    The NumPy interpreter is a format-level check, not a substitute for ncnn/HF
    real-checkpoint parity. Run: python tests/test_minicpm.py vision-export.
    """
    import importlib.util
    import json
    import math
    import os
    import subprocess
    from pathlib import Path
    import tempfile
    import unittest

    import numpy as np

    spec = importlib.util.spec_from_file_location("vision_export", Path(__file__).parents[1] / "export/minicpm_v47_vision_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


    def config():
        return {"model_type": "minicpmv4_7", "insert_layer_id": 0,
                "text_config": {"hidden_size": 6},
                "vision_config": {"hidden_size": 8, "intermediate_size": 12,
                                  "num_attention_heads": 2, "num_hidden_layers": 2,
                                  "image_size": 8, "patch_size": 2, "num_channels": 3,
                                  "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6}}


    class Source:
        def __init__(self, graphs):
            rng = np.random.default_rng(413)
            self.values = {}
            self.calls = []
            for g in graphs.values():
                for key, shape, _ in g.weights:
                    if key not in self.values:
                        self.values[key] = (rng.standard_normal(shape) * .1).astype(np.float32)
                        if "norm" in key and key.endswith("weight"):
                            self.values[key] += 1
            self.values[module.VISION + ".embeddings.position_embedding.weight"] = rng.standard_normal((16, 8)).astype(np.float32)

        def get(self, key):
            self.calls.append(key)
            return self.values[key]


    def norm(x, weight, bias, eps):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * weight + bias


    def gelu(x, tanh):
        if tanh:
            return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))
        return .5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


    def interpret(param, binary, value):
        """Evaluate exported bytes using ncnn's dimension and weight-tag conventions."""
        data = memoryview(binary)
        offset = 0
        blobs = {"in0": value}

        def read(shape, tagged=False):
            nonlocal offset
            if tagged:
                assert bytes(data[offset:offset + 4]) == b"\0\0\0\0"
                offset += 4
            size = math.prod(shape)
            result = np.frombuffer(data[offset:offset + size * 4], dtype="<f4").reshape(shape)
            offset += size * 4
            return result

        lines = param.splitlines()
        assert int(lines[1].split()[0]) == len(lines) - 2
        for line in lines[2:]:
            fields = line.split()
            kind, _, ni, no = fields[:4]
            ni, no = int(ni), int(no)
            inputs = fields[4:4 + ni]
            outputs = fields[4 + ni:4 + ni + no]
            pd = dict(p.split("=", 1) for p in fields[4 + ni + no:])
            if kind == "Input":
                continue
            x = blobs[inputs[0]]
            if kind == "Split":
                for out in outputs:
                    blobs[out] = x
                continue
            if kind == "LayerNorm":
                n = int(pd["0"])
                y = norm(x, read((n,)), read((n,)), float(pd["1"]))
            elif kind == "Gemm":
                n, k = int(pd["8"]), int(pd["9"])
                y = x @ read((n, k), True).T + read((n,), True)
            elif kind == "Reshape":
                dims = [int(pd[k]) for k in ("2", "1", "0") if k in pd]
                y = x.reshape(dims)
            elif kind == "Permute":
                order = int(pd["0"])
                y = x.T if x.ndim == 2 and order == 1 else x.transpose(1, 0, 2)
            elif kind == "SDPA":
                q, k, v = (blobs[name] for name in inputs)
                scores = q @ k.transpose(0, 2, 1) * float(pd["6"])
                scores = np.exp(scores - scores.max(-1, keepdims=True))
                y = (scores / scores.sum(-1, keepdims=True)) @ v
            elif kind == "BinaryOp":
                y = x + blobs[inputs[1]]
            elif kind == "GELU":
                y = gelu(x, bool(int(pd.get("0", "0"))))
            elif kind == "Noop":
                y = x
            elif kind == "Convolution":
                n, p = int(pd["0"]), int(pd["1"])
                w = read((n, x.shape[0], p, p), True)
                b = read((n,))
                patches = np.lib.stride_tricks.sliding_window_view(x, (p, p), axis=(1, 2))[:, ::p, ::p]
                y = np.einsum("chwkl,ockl->ohw", patches, w) + b[:, None, None]
            else:
                raise AssertionError(kind)
            blobs[outputs[0]] = y
        assert offset == len(binary), (offset, len(binary))
        return blobs["out0"]


    def attention_reference(x, state, base, heads=2):
        ln = norm(x, state[base + ".layer_norm1.weight"], state[base + ".layer_norm1.bias"], 1e-6)
        qkv = []
        for c in "qkv":
            key = base + ".self_attn." + c + "_proj"
            qkv.append((ln @ state[key + ".weight"].T + state[key + ".bias"]).reshape(len(x), heads, -1))
        q, k, v = qkv
        scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(q.shape[-1])
        prob = np.exp(scores - scores.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        y = np.einsum("hts,shd->thd", prob, v).reshape(x.shape)
        base += ".self_attn.out_proj"
        return x + y @ state[base + ".weight"].T + state[base + ".bias"]


    def block_reference(x, state, base):
        x = attention_reference(x, state, base)
        y = norm(x, state[base + ".layer_norm2.weight"], state[base + ".layer_norm2.bias"], 1e-6)
        y = gelu(y @ state[base + ".mlp.fc1.weight"].T + state[base + ".mlp.fc1.bias"], True)
        return x + y @ state[base + ".mlp.fc2.weight"].T + state[base + ".mlp.fc2.bias"]


    class VisionExportTests(unittest.TestCase):
        def setUp(self):
            self.graphs = module.build_graphs(config())
            self.source = Source(self.graphs)
            self.tmp = tempfile.TemporaryDirectory()
            self.addCleanup(self.tmp.cleanup)
            self.path = Path(self.tmp.name)

        def run_graph(self, name, x):
            files = self.graphs[name].write(self.source, self.path, name)
            return interpret((self.path / files["param"]).read_text(), (self.path / files["bin"]).read_bytes(), x)

        def test_encoder_components_numerical(self):
            x = np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32)
            expected = block_reference(x, self.source.values, module.VISION + ".encoder.layers.0")
            np.testing.assert_allclose(self.run_graph("encoder_pre", x), expected, atol=2e-6, rtol=2e-6)
            y = block_reference(x, self.source.values, module.VISION + ".encoder.layers.1")
            key = module.VISION + ".post_layernorm"
            expected = norm(y, self.source.values[key + ".weight"], self.source.values[key + ".bias"], 1e-6)
            np.testing.assert_allclose(self.run_graph("encoder_post", x), expected, atol=2e-6, rtol=2e-6)

        def test_window_attention_numerical(self):
            x = np.random.default_rng(3).standard_normal((4, 8)).astype(np.float32)
            expected = attention_reference(x, self.source.values, module.VISION + ".vit_merger")
            np.testing.assert_allclose(self.run_graph("window_attn", x), expected, atol=2e-6, rtol=2e-6)

        def test_projectors_numerical_and_activation(self):
            x = np.random.default_rng(5).standard_normal((3, 32)).astype(np.float32)
            for name, base, tanh in (("window_mlp", module.VISION + ".vit_merger", True),
                                     ("projector", "model.merger.mlp.0", False)):
                s = self.source.values
                y = norm(x, s[base + ".pre_norm.weight"], s[base + ".pre_norm.bias"], 1e-6)
                y = gelu(y @ s[base + ".linear_1.weight"].T + s[base + ".linear_1.bias"], tanh)
                expected = y @ s[base + ".linear_2.weight"].T + s[base + ".linear_2.bias"]
                np.testing.assert_allclose(self.run_graph(name, x), expected, atol=2e-6, rtol=2e-6)

        def test_patch_convolution_row_order(self):
            x = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) / 100
            s = self.source.values
            key = module.VISION + ".embeddings.patch_embedding"
            rows = []
            for y in range(0, 8, 2):
                for z in range(0, 8, 2):
                    rows.append((s[key + ".weight"] * x[None, :, y:y + 2, z:z + 2]).sum((1, 2, 3)) + s[key + ".bias"])
            np.testing.assert_allclose(self.run_graph("patch_embed", x), rows, atol=2e-6)

        def test_windows_and_positions(self):
            np.testing.assert_array_equal(window_indices(4, 4), [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
            np.testing.assert_array_equal(nearest_position_ids(2, 2, 4), [0, 2, 8, 10])
            with self.assertRaises(ValueError):
                window_indices(3, 4)

        def test_manifest_and_weight_shape_guard(self):
            (self.path / "config.json").write_text(json.dumps(config()))
            out = self.path / "out"
            manifest = module.export_minicpm_v47_vision(self.path, out, source=self.source)
            self.assertEqual(set(manifest["components"]), set(self.graphs))
            self.assertEqual((out / manifest["position_embeddings"]["file"]).stat().st_size, 16 * 8 * 4)
            self.assertEqual(manifest["output_hidden_size"], 6)
            first = self.graphs["window_mlp"].weights[0][0]
            self.source.values[first] = np.zeros(1)
            with self.assertRaisesRegex(ValueError, "expected"):
                self.graphs["window_mlp"].write(self.source, self.path, "broken")
            self.assertFalse((self.path / "broken.ncnn.bin").exists())
            self.assertFalse((self.path / "broken.ncnn.bin.tmp").exists())

        def test_native_ncnn_component_parity_when_probe_available(self):
            probe = os.environ.get("NCNN_COMPONENT_PROBE")
            if not probe:
                self.skipTest("Set NCNN_COMPONENT_PROBE to the compiled C++ probe")
            rng = np.random.default_rng(19)
            shapes = {"patch_embed": (3, 8, 8), "encoder_pre": (16, 8),
                      "encoder_post": (4, 8), "window_attn": (4, 8),
                      "window_mlp": (4, 32), "projector": (1, 32)}
            for name, shape in shapes.items():
                with self.subTest(component=name):
                    x = rng.standard_normal(shape).astype(np.float32)
                    expected = self.run_graph(name, x)
                    input_path, output_path = self.path / "input.f32", self.path / "output.f32"
                    x.tofile(input_path)
                    result = subprocess.run([probe, str(self.path / (name + ".ncnn.param")),
                        str(self.path / (name + ".ncnn.bin")), str(input_path), str(output_path),
                        str(len(shape)), str(shape[-1]), str(shape[-2]),
                        str(shape[0] if len(shape) == 3 else 1)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual_shape = tuple(map(int, result.stdout.strip().split()))
                    self.assertEqual(actual_shape, expected.shape)
                    actual = np.fromfile(output_path, dtype=np.float32).reshape(actual_shape)
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

        def test_ncnn_component_parity_when_bindings_available(self):
            try:
                import ncnn
            except ImportError:
                self.skipTest("ncnn Python binding is not installed")
            if not hasattr(ncnn, "Net"):
                self.skipTest("ncnn source namespace is present, but Python binding is not installed")
            rng = np.random.default_rng(9)
            shapes = {"patch_embed": (3, 8, 8), "encoder_pre": (16, 8),
                      "encoder_post": (4, 8), "window_attn": (4, 8),
                      "window_mlp": (4, 32), "projector": (1, 32)}
            for name, shape in shapes.items():
                with self.subTest(component=name):
                    x = rng.standard_normal(shape).astype(np.float32)
                    expected = self.run_graph(name, x)
                    net = ncnn.Net()
                    net.opt.use_vulkan_compute = False
                    net.opt.use_fp16_storage = False
                    net.opt.use_fp16_arithmetic = False
                    net.opt.use_bf16_storage = False
                    net.opt.use_packing_layout = False
                    net.opt.num_threads = 1
                    self.assertEqual(net.load_param(str(self.path / (name + ".ncnn.param"))), 0)
                    self.assertEqual(net.load_model(str(self.path / (name + ".ncnn.bin"))), 0)
                    ex = net.create_extractor()
                    owned_input = x.copy()
                    input_mat = ncnn.Mat(owned_input).clone()
                    self.assertEqual(ex.input("in0", input_mat), 0)
                    status, out = ex.extract("out0")
                    self.assertEqual(status, 0)
                    np.testing.assert_allclose(np.array(out), expected, atol=2e-5, rtol=2e-5)

        def test_bfloat16_lazy_source_when_torch_available(self):
            try:
                import torch
                from safetensors.torch import save_file
            except ImportError:
                self.skipTest("torch/safetensors are not installed")
            tensor = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
            save_file({"sample": tensor}, str(self.path / "model.safetensors"))
            source = module.SafeTensorSource(self.path)
            self.assertEqual(source.get("sample").dtype, np.float32)
            np.testing.assert_array_equal(source.get("sample"), tensor.float().numpy())
            with self.assertRaisesRegex(KeyError, "Missing"):
                source.get("not_present")

        def test_full_architecture_and_streaming_descriptors(self):
            cfg = config()
            cfg["insert_layer_id"] = 6
            cfg["vision_config"]["num_hidden_layers"] = 27
            graphs = module.build_graphs(cfg)
            self.assertEqual(graphs["encoder_pre"].param().count("SDPA "), 7)
            self.assertEqual(graphs["encoder_post"].param().count("SDPA "), 20)
            self.assertNotIn("RotaryEmbed", graphs["encoder_pre"].param())
            for g in graphs.values():
                for key, shape, tagged in g.weights:
                    self.assertIsInstance(key, str)
                    self.assertIsInstance(shape, tuple)
                    self.assertIsInstance(tagged, bool)
            cfg["merger_times"] = 2
            with self.assertRaises(ValueError):
                module.build_graphs(cfg)

        def test_ncnn_float_parser_precision(self):
            # Mirror ncnn ParamDict's unsigned 32-bit fraction accumulators.
            def ncnn_float(value):
                mantissa, _, exponent = value.lower().partition("e")
                integer, _, fraction = mantissa.partition(".")
                numerator, denominator = 0, 1
                for digit in fraction:
                    numerator = (numerator * 10 + int(digit)) & 0xffffffff
                    denominator = (denominator * 10) & 0xffffffff
                return np.float32((int(integer) + numerator / denominator)
                                  * 10 ** int(exponent or "0"))

            self.assertGreater(abs(float(ncnn_float("0.117851130198"))
                                   - 1 / math.sqrt(72)), .4)
            cfg = config()
            cfg["vision_config"].update(hidden_size=144, intermediate_size=192)
            for graph in module.build_graphs(cfg).values():
                for line in graph.layers:
                    for field in line.split():
                        if "=" not in field:
                            continue
                        value = field.split("=", 1)[1]
                        if "." not in value and "e" not in value.lower():
                            continue
                        self.assertLessEqual(len(value), 15)
                        self.assertLessEqual(len(value.split("e")[0].split(".")[-1]), 8)
                        self.assertEqual(ncnn_float(value), np.float32(float(value)))
                    if line.startswith("SDPA "):
                        value = line.split("6=", 1)[1]
                        self.assertEqual(ncnn_float(value), np.float32(1 / math.sqrt(72)))

        def test_ncnn_head_dim_72_regression(self):
            try:
                import ncnn
            except ImportError:
                self.skipTest("ncnn Python binding is not installed")
            if not hasattr(ncnn, "Net"):
                self.skipTest("ncnn Python binding is not installed")
            cfg = config()
            cfg["vision_config"].update(hidden_size=144, intermediate_size=192)
            self.graphs = module.build_graphs(cfg)
            self.source = Source(self.graphs)
            x = np.random.default_rng(72).standard_normal((7, 144)).astype(np.float32)
            expected = self.run_graph("encoder_pre", x)
            net = ncnn.Net()
            net.opt.use_vulkan_compute = False
            net.opt.use_fp16_storage = False
            net.opt.use_fp16_arithmetic = False
            net.opt.use_bf16_storage = False
            net.opt.use_packing_layout = False
            net.opt.num_threads = 1
            self.assertEqual(net.load_param(str(self.path / "encoder_pre.ncnn.param")), 0)
            self.assertEqual(net.load_model(str(self.path / "encoder_pre.ncnn.bin")), 0)
            ex = net.create_extractor()
            owned_input = x.copy()
            input_mat = ncnn.Mat(owned_input).clone()
            self.assertEqual(ex.input("in0", input_mat), 0)
            status, output = ex.extract("out0")
            self.assertEqual(status, 0)
            np.testing.assert_allclose(np.array(output), expected, atol=2e-5, rtol=2e-5)

        def test_native_ncnn_head_dim_72_regression(self):
            probe = os.environ.get("NCNN_COMPONENT_PROBE")
            if not probe:
                self.skipTest("Set NCNN_COMPONENT_PROBE to the compiled C++ probe")
            cfg = config()
            cfg["vision_config"].update(hidden_size=144, intermediate_size=192)
            self.graphs = module.build_graphs(cfg)
            self.source = Source(self.graphs)
            x = np.random.default_rng(72).standard_normal((7, 144)).astype(np.float32)
            expected = self.run_graph("encoder_pre", x)
            input_path, output_path = self.path / "input.f32", self.path / "output.f32"
            x.tofile(input_path)
            result = subprocess.run([probe, str(self.path / "encoder_pre.ncnn.param"),
                str(self.path / "encoder_pre.ncnn.bin"), str(input_path), str(output_path),
                "2", "144", "7", "1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(tuple(map(int, result.stdout.strip().split())), expected.shape)
            actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

        def test_native_full_runtime_owned_window_regression(self):
            probe = os.environ.get("NCNN_VISION_PROBE")
            if not probe:
                self.skipTest("Set NCNN_VISION_PROBE to the vision-only C++ probe")
            from PIL import Image
            cfg = config()
            cfg["vision_config"].update(patch_size=14, image_size=56)
            self.graphs = module.build_graphs(cfg)
            self.source = Source(self.graphs)
            (self.path / "config.json").write_text(json.dumps(cfg))
            model_dir = self.path / "model"
            module.export_minicpm_v47_vision(self.path, model_dir, source=self.source)
            vision = {"manifest": "minicpm_v47_vision.json", "scale_resolution": 56,
                      "slice_mode": False, "max_slice_nums": 1,
                      "marker_ids": dict(zip(("image", "image_start", "image_end",
                                               "slice_start", "slice_end", "newline"), range(1, 7)))}
            (model_dir / "model.json").write_text(json.dumps({"setting": {"vision": vision}}))
            rgb = (np.arange(56 * 56 * 3, dtype=np.uint16) % 256).astype(np.uint8).reshape(56, 56, 3)
            image_path, output_path = self.path / "fixture.png", self.path / "features.f32"
            Image.fromarray(rgb).save(image_path)
            pixels = (rgb.transpose(2, 0, 1).astype(np.float32) / 255 - .5) / .5
            x = self.run_graph("patch_embed", pixels)
            x += self.source.values[module.VISION + ".embeddings.position_embedding.weight"][
                nearest_position_ids(4, 4, 4)]
            x = self.run_graph("encoder_pre", x)
            windows = x[window_indices(4, 4)].reshape(-1, 4, 8)
            attended = np.stack([self.run_graph("window_attn", window) for window in windows])
            x = self.run_graph("window_mlp", attended.reshape(-1, 32)) + attended.mean(axis=1)
            x = self.run_graph("encoder_post", x)
            expected = self.run_graph("projector", x[window_indices(2, 2)].reshape(-1, 32))
            result = subprocess.run([probe, str(model_dir), str(image_path), str(output_path), "1"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            shape = json.loads(result.stdout)
            self.assertEqual((shape["rows"], shape["cols"]), expected.shape)
            self.assertFalse(shape["language_model_loaded"])
            actual = np.fromfile(output_path, dtype=np.float32).reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

        def test_readonly_parity_rejects_corrupt_copy(self):
            import shutil
            from unittest.mock import patch
            cfg = config()
            model_dir = self.path / 'original'
            (self.path / 'config.json').write_text(json.dumps(cfg))
            manifest = module.export_minicpm_v47_vision(self.path, model_dir, source=self.source)
            pixels = np.random.default_rng(71).standard_normal((3, 8, 8)).astype(np.float32)
            with ReadOnlyVisionPackage(model_dir) as package:
                expected = evaluate_vision_package(package, pixels, 1)
                hashes = package.before
            corrupt = self.path / 'corrupt'
            shutil.copytree(model_dir, corrupt)
            # Destroy only the copied final projector weight/bias floats while
            # retaining their ncnn tags and shape; this must not be regenerated.
            weights = corrupt / manifest['components']['projector']['bin']
            values = np.fromfile(weights, dtype='<f4')
            values[-cfg['text_config']['hidden_size']:] += 100
            values.tofile(weights)
            with patch.object(module, 'export_minicpm_v47_vision', side_effect=AssertionError('parity must never export')):
                with ReadOnlyVisionPackage(corrupt) as package:
                    actual = evaluate_vision_package(package, pixels, 1)
                    with self.assertRaises(AssertionError):
                        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-4)
            self.assertEqual(ReadOnlyVisionPackage(model_dir).before, hashes)
    return types.SimpleNamespace(**locals())


def _vision_reference():
    """Actual Transformers reference parity for independent C++ MiniCPM helpers.

    No checkpoint tensors or language/vision modules are allocated. Uses the real
    MiniCPMV4_7Model positioning methods and Qwen3_5TextRotaryEmbedding from installed
    Transformers; config is read from the requested checkpoint directory.
    """
    import argparse
    import pathlib
    import subprocess

    import numpy as np
    import torch
    from transformers import AutoConfig
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7Model
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from transformers.vision_utils import get_vision_nearest_position_ids


    def main():
        torch.set_num_threads(1)
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-dir', required=True)
        args = parser.parse_args()
        cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
        # Only pure positioning methods need config. Bypass constructor intentionally
        # to avoid allocating either checkpoint-sized or random surrogate networks.
        model = MiniCPMV4_7Model.__new__(MiniCPMV4_7Model)
        torch.nn.Module.__init__(model)
        model.config = cfg
        model.rope_deltas = None
        rotary = Qwen3_5TextRotaryEmbedding(cfg.text_config)
        params = cfg.text_config.rope_parameters
        dim = int(cfg.text_config.head_dim * params['partial_rotary_factor'])
        markers = [cfg.image_token_id, cfg.image_start_id, cfg.image_end_id,
                   cfg.slice_start_id, cfg.slice_end_id, cfg.newline_id]
        side = cfg.vision_config.image_size // cfg.vision_config.patch_size
        cases = []
        for mode in ('16x', '4x'):
            merge = 4 if mode == '16x' else 2
            for h,w in ((32,32), (36,28), (28,36), (4,4), (8,12), (12,8), (68,72)):
                for base in (0,17,513):
                    for newlines in (0,1,3):
                        for suffix in ([], [41,42]):
                            ids = [31,32] + [cfg.image_start_id] + [cfg.image_token_id]*(h*w//merge**2)
                            ids += [cfg.image_end_id]+[cfg.newline_id]*newlines+suffix
                            cases.append((mode,h,w,base,ids))
        records = []
        for mode,h,w,base,ids in cases:
            fields = [mode,h,w,side,base,dim,params['rope_theta'],*markers,*params['mrope_section'],len(ids),*ids]
            records.append(' '.join(map(str,fields)))
        probe = probe_path('minicpm_vision_reference_probe', 'executable')
        result = subprocess.run([str(probe)],input='\n'.join(records)+'\n',capture_output=True,text=True,check=True)
        lines = result.stdout.splitlines()
        assert len(lines) == 5*len(cases)
        max_rope_error = 0.0
        for i,(mode,h,w,base,ids) in enumerate(cases):
            label = (mode,h,w,base,len(ids))
            delta,next_pos = map(int,lines[i*5].split())
            actual = np.fromstring(lines[i*5+1],sep=' ',dtype=np.int64).reshape(-1,3)
            nearest = np.fromstring(lines[i*5+2],sep=' ',dtype=np.int64)
            cc = np.fromstring(lines[i*5+3],sep=' ',dtype=np.float32).reshape(-1,dim//2)
            ss = np.fromstring(lines[i*5+4],sep=' ',dtype=np.float32).reshape(-1,dim//2)
            # A separately cached system prefix is represented by `base` plain text
            # tokens in the full HF reference. Check only the current suffix region.
            full_ids = torch.tensor([[77]*base+ids],dtype=torch.long)
            mm = (full_ids == cfg.image_token_id).int()
            ref_pos,ref_delta = model.get_rope_index(full_ids,mm_token_type_ids=mm,
                               target_sizes=torch.tensor([[h,w]]),downsample_mode=mode)
            expected = ref_pos[:,0,base:].T.numpy()
            np.testing.assert_array_equal(actual,expected,err_msg=str(label))
            # Helper delta is relative to suffix length. Cached text contributes base
            # to that delta; the canonical HF full-sequence delta excludes it.
            assert delta-base == ref_delta.item(), (label,delta,ref_delta)
            assert next_pos == full_ids.shape[-1]+ref_delta.item(), (label,next_pos,ref_delta)
            nearest_ref = get_vision_nearest_position_ids(torch.tensor([[h,w]]),side).numpy()
            np.testing.assert_array_equal(nearest,nearest_ref,err_msg=str(label))
            with torch.no_grad():
                c,s = rotary(torch.zeros(1,len(ids),1),ref_pos[:,:,base:])
            for out,ref in ((cc,c[0,:,:dim//2].numpy()),(ss,s[0,:,:dim//2].numpy())):
                max_rope_error = max(max_rope_error,float(np.max(np.abs(out-ref))))
                np.testing.assert_allclose(out,ref,atol=7e-5,rtol=1e-5,err_msg=str(label))
            # Official cached generation method must continue from next_position.
            model.rope_deltas = ref_delta
            class CacheLength:
                def get_seq_length(self): return full_ids.shape[-1]
            decoded = model.compute_3d_position_ids(None,inputs_embeds=torch.zeros(1,3,1),
                                                    past_key_values=CacheLength())
            np.testing.assert_array_equal(decoded[:,0].numpy(),
                                          np.tile(np.arange(next_pos,next_pos+3),(3,1)),err_msg=str(label))
        print(f'PASS: {len(cases)} real HF canvas/nearest/RoPE/cached-decode comparisons; max RoPE abs error={max_rope_error:.9g}')
        print('No model weights loaded; this establishes helper numerical parity only.')
    return types.SimpleNamespace(**locals())


# Historical measurements (not assertions about a new run).
HISTORICAL_RESULTS = {'minicpm_v47_refactor_validation': {'date_utc': '2026-10-14',
                                     'source_base_commit': '09fb5f3312e63b9abdd8b803c0548303a0484042',
                                     'model': 'openbmb/MiniCPM-V-4.7-1B',
                                     'model_revision': 'ef3ab3bf0120a60e1a2d2b52d984a79d8cb7c630',
                                     'ncnn_revision': 'b0f97f43f266617ff5e5096d440dbe3f9a4c35b8',
                                     'platform': 'Linux x86_64 CPU FP32',
                                     'existing_converted_package_reused': True,
                                     'vision_threads': 2,
                                     'vqa_threads': 4,
                                     'vqa_max_new_tokens': 32,
                                     'prompt': 'Describe this image in one short sentence.',
                                     'tests': {'ctest_passed': 7,
                                               'legacy_optional_model_cases_skipped': 3,
                                               'legacy_skip_reason': 'Optional original text/int8 model '
                                                                     'packages absent; shared image/vision '
                                                                     'utilities were tested directly',
                                               'vision_export_unit_tests_passed': 14,
                                               'vision_export_skipped': 0,
                                               'pillow_byte_exact_resize_cases': 1290,
                                               'official_pil_full_preprocess_cases': 20,
                                               'official_pil_preprocess_max_abs': 0.0,
                                               'hf_canvas_nearest_rope_cases': 252,
                                               'hf_rope_max_abs': 1.57281756e-05},
                                     'runtime_source_sha256': {'CMakeLists.txt': '0c482c4d34344db6e1ea6078d1f141ae231ea78f7b92eab3d71e235dc190e3e1',
                                                               'src/ncnn_llm_gpt.cpp': '9cf786aee64366e03898616b5a5b938985039e3839db70b2bf2295025d84b892',
                                                               'src/ncnn_llm_gpt.h': 'ebfd07cfb28f493bf63a2eb9c30a228bce6958561dd244efceed8f089b0c493a',
                                                               'src/utils/image_utils.cpp': 'bc6fcb98fef1bab7c2f99051e9fcf1deb935cdbc580ec723959dffb1b51cd6cb',
                                                               'src/utils/image_utils.h': '6e04200c6f421e2b631f5b550ce4dd0153eee76ced3d1db99c098435e34a9123',
                                                               'src/utils/rope_embed.cpp': '9453e83328bfeb56a0964525281e650f77de566cc6499809e9c0d6896f379c8e',
                                                               'src/utils/rope_embed.h': '639b37819ce04b527934ef726cd8dc34a9bfef586100505446de18fee05f0818'},
                                     'trained_vision': {'returncode': 0,
                                                        'seconds': 51.28915585699724,
                                                        'shape': [64, 1024],
                                                        'hf_max_abs': 6.937980651855469e-05,
                                                        'hf_mean_abs': 1.4625606682555636e-06,
                                                        'hf_cosine': 0.9999999999062039,
                                                        'pre_refactor_max_abs': 0.0,
                                                        'bitwise_equal_to_pre_refactor': True},
                                     'trained_decoder': {'returncode': 0,
                                                         'seconds': 5.725434425010462,
                                                         'trained_decoder': True,
                                                         'full_vqa': False,
                                                         'shape': [1024, 8],
                                                         'max_abs': 0.0004215240478515625,
                                                         'mean_abs': 1.2515714690408686e-05,
                                                         'cosine': 0.9999999999891479,
                                                         'relative_l2': 4.6627328734838095e-06,
                                                         'atol': 0.0005,
                                                         'rtol': 0.0005,
                                                         'passed': True},
                                     'trained_decoder_parity': {'returncode': 0,
                                                                'seconds': 0.21527768499800004},
                                     'refract-448': {'returncode': 0,
                                                     'seconds': 37.911172776002786,
                                                     'image': 'refract-448.png',
                                                     'text': 'A red pencil with a yellow tip rests in a '
                                                             'glass of water, partially submerged.',
                                                     'pre_refactor_text': 'A red pencil with a yellow tip '
                                                                          'rests in a glass of water, '
                                                                          'partially submerged.',
                                                     'exact_match': True,
                                                     'hf_reference_available': True},
                                     'bee-448': {'returncode': 0,
                                                 'seconds': 36.86429378899629,
                                                 'image': 'bee-448.png',
                                                 'text': 'A bumblebee rests on a pink cosmos flower, '
                                                         'surrounded by wilting blooms and red flowers in a '
                                                         'garden.',
                                                 'pre_refactor_text': 'A bumblebee rests on a pink cosmos '
                                                                      'flower, surrounded by wilting blooms '
                                                                      'and red flowers in a garden.',
                                                 'exact_match': True,
                                                 'hf_reference_available': False},
                                     'max_child_rss_kib': 6442196,
                                     'limitations': ['Single-image 448x448 thumbnails,16x,CPU FP32 only',
                                                     'No full legacy Qwen/ViT checkpoint regression: weights '
                                                     'unavailable',
                                                     'Bee checked against pre-refactor answer; no '
                                                     'independent HF bee reference',
                                                     'No '
                                                     'Windows/macOS/GPU/quantized/detail-slice/video/multi-image '
                                                     'validation'],
                                     'published_commit': 'a14fe16f49ea6ad36767ade62bba9d6d3d1d8e60'},
 'minicpm_v47_tokenizer_validation': {'case_count': 6549,
                                      'failure_count': 0,
                                      'failures': [],
                                      'reference': 'actual '
                                                   'tokenizers.Tokenizer.from_file(checkpoint/tokenizer.json)',
                                      'invalid_utf8_rejected': True,
                                      'unsupported_regex_rejected': True,
                                      'invalid_artifact_cases': ['regex',
                                                                 'missing_end',
                                                                 'source_hex',
                                                                 'decomposition_count',
                                                                 'added_vocab_mismatch',
                                                                 'excess_fields',
                                                                 'duplicate_R',
                                                                 'duplicate_D',
                                                                 'duplicate_C',
                                                                 'duplicate_P',
                                                                 'duplicate_A'],
                                      'unsupported_pipeline_cases': 4},
 'minicpm_v47_vision_validation': {'embedding': {'shape': [1024, 1152],
                                                 'max_abs_error': 7.3909759521484375e-06,
                                                 'mean_abs_error': 1.4801309058645432e-07,
                                                 'cosine_similarity': 0.9999999999999065},
                                   'pre': {'shape': [1024, 1152],
                                           'max_abs_error': 6.914138793945312e-05,
                                           'mean_abs_error': 1.5024428581379546e-06,
                                           'cosine_similarity': 0.9999999999859541},
                                   'window': {'shape': [256, 1152],
                                              'max_abs_error': 0.000270843505859375,
                                              'mean_abs_error': 1.4696346044315887e-06,
                                              'cosine_similarity': 0.9999999999286769},
                                   'post': {'shape': [256, 1152],
                                            'max_abs_error': 0.0003414154052734375,
                                            'mean_abs_error': 2.5967020255023165e-06,
                                            'cosine_similarity': 0.999999999984749},
                                   'features': {'shape': [64, 1024],
                                                'max_abs_error': 8.058547973632812e-05,
                                                'mean_abs_error': 1.5337180334570877e-06,
                                                'cosine_similarity': 0.9999999999039767},
                                   'metadata': {'ncnn_version': '1.0.20260526',
                                                'image': 'refract-448.png',
                                                'dtype': 'float32',
                                                'input_buffer_retained': True,
                                                'model_repository': 'openbmb/MiniCPM-V-4.7-1B',
                                                'model_revision': 'ef3ab3bf0120a60e1a2d2b52d984a79d8cb7c630',
                                                'checkpoint': {'name': 'model.safetensors',
                                                               'bytes': 3109054480,
                                                               'sha256': '8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002'},
                                                'fixture': {'source_url': 'https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png',
                                                            'original_size': [406, 528],
                                                            'original_sha256': 'e384f22e7a10381084d3dafba477cf0fa59fa671612027e87e675de5ad149441',
                                                            'test_fixture': 'refract-448.png',
                                                            'test_fixture_sha256': '969dbe26ad4dd2d94daad99b6b7c172ea7adf9f56305ee416d39794b9fbd299a',
                                                            'processing': 'Pillow ImageOps.pad RGB448x448 '
                                                                          'bicubic, background240; both '
                                                                          'reference and ncnn receive the '
                                                                          'same saved fixture'},
                                                'libraries': {'numpy': '2.5.2',
                                                              'pillow': '12.3.0',
                                                              'transformers': '5.19.0',
                                                              'torch': '2.14.1+cpu',
                                                              'ncnn': '1.0.20260526',
                                                              'safetensors': '0.8.0'},
                                                'platform': 'Linux CPU',
                                                'downsample_mode': '16x',
                                                'source_checkpoint_dtype': 'BF16',
                                                'exported_weight_dtype': 'FP32',
                                                'reference': 'Official Transformers MiniCPMV4_7Model with '
                                                             'only vision_tower and merger materialized; '
                                                             'language parameters remained meta',
                                                'patch_packing_proof': {'processor_pixel_shape': [1,
                                                                                                  3,
                                                                                                  14,
                                                                                                  14336],
                                                                        'target_sizes': [[32, 32]],
                                                                        'patch_element_order_equivalent': True,
                                                                        'pixel_max_abs_error': 5.9138983488082886e-08,
                                                                        'embedding_max_abs_error': 7.152557373046875e-07,
                                                                        'embedding_mean_abs_error': 9.873195594423123e-09,
                                                                        'embedding_cosine_similarity': 0.9999999999999983},
                                                'acceptance': {'atol': 0.001,
                                                               'rtol': 0.0001,
                                                               'min_cosine': 0.999999},
                                                'scope': 'One real 448x448 thumbnail; '
                                                         'multi-slice/video/language generation are not '
                                                         'covered by this report',
                                                'parameter_float_format': '.8e',
                                                'arrays': {'hf_embedding.npy': {'bytes': 4718720,
                                                                                'sha256': '3bc28348dc428f8544cd2d424b7db56930cef310b4d9662494fef002645ed493'},
                                                           'hf_features.npy': {'bytes': 262272,
                                                                               'sha256': '940346f3ebfd50c06dd40cf568fdc50dd70583424509222d08b7ff966c316d07'},
                                                           'hf_post.npy': {'bytes': 1179776,
                                                                           'sha256': '3486756737cf43514ab64103fac045cdd4c5989540d35a36f784f28cc89e0eb3'},
                                                           'hf_pre.npy': {'bytes': 4718720,
                                                                          'sha256': '9cd916c08bbf45c56ccae5dd818788c15efea40b0207a638e827dde6c96fe350'},
                                                           'hf_window.npy': {'bytes': 1179776,
                                                                             'sha256': 'a98594606bbe85777d43a95f2c982732ed72b62a820fc634c59c28291391d851'},
                                                           'ncnn_embedding.npy': {'bytes': 4718720,
                                                                                  'sha256': '219d104dcc47075f00f2307a7caec2f22fe56dd52ed1bef61e51207e1e822837'},
                                                           'ncnn_features.npy': {'bytes': 262272,
                                                                                 'sha256': '1e76fab873e2db26b550e2770aaf0e3d2008e4db7a1993953d685a77a605cc1f'},
                                                           'ncnn_post.npy': {'bytes': 1179776,
                                                                             'sha256': '81a27d4efb446ef5fbf1650c325505a06a6bede045a4bb3038078497a8bdec08'},
                                                           'ncnn_pre.npy': {'bytes': 4718720,
                                                                            'sha256': '9591947b78cf5ee70925d90508eab17cdc8daa473a5d4c9baaca81c0fc822995'},
                                                           'ncnn_pre_exact_input.npy': {'bytes': 4718720,
                                                                                        'sha256': '2449ec594791827ed01ac5fb672ae33b57f8745088e44063054c3c9bd4079a53'},
                                                           'ncnn_window.npy': {'bytes': 1179776,
                                                                               'sha256': '36f9da27b24b0f020230c10e458c70601f280e659efcf5bb63d3ae305777c358'}},
                                                'parameter_files': {'minicpm-converted_vision_encoder_post.ncnn.param': {'sha256': '596ea1d0a41cf4bfb64098a0f715e1aa98b4db58f653db0bcad2c246aafc6526',
                                                                                                                         'bytes': 24415},
                                                                    'minicpm-converted_vision_encoder_pre.ncnn.param': {'sha256': '0c2201588f0d9b48ee4bbb1376fa864e09b27e839cbab9caecdbfc4e11aae3e4',
                                                                                                                        'bytes': 8307},
                                                                    'minicpm-converted_vision_patch_embed.ncnn.param': {'sha256': 'ded83bed0937fd664b1ca5a96587225793d8185f17bcf92770251f14c9d6fec5',
                                                                                                                        'bytes': 239},
                                                                    'minicpm-converted_vision_projector.ncnn.param': {'sha256': '8207f504efaf55547d61903061770145dc6c4018ec8ba4ea11f54bc94add884b',
                                                                                                                      'bytes': 265},
                                                                    'minicpm-converted_vision_window_attn.ncnn.param': {'sha256': '58017535c538be060e06e5bc15dc5faa70ebdd7b26b1ffbab209ff03a08902ef',
                                                                                                                        'bytes': 856},
                                                                    'minicpm-converted_vision_window_mlp.ncnn.param': {'sha256': '1aea6a5339ea6c99233b565578e89171d4615e2d41fed03aec059991565a1e17',
                                                                                                                       'bytes': 267}},
                                                'exporter_sha256': 'b8f29e62aeb7311dd46e53b5c4e35f9d81630f650b4edc0d88a2971cbd034eed',
                                                'official_modeling_sha256': '2ac52bbe1615eb1d02cf4cce981ed4d20a9dc11116ee50bb92d0a2aaf227dd42',
                                                'official_vision_utils_sha256': 'bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356',
                                                'encoder_pre_exact_hf_input_max_abs_error': 7.009506225585938e-05,
                                                'native_synthetic_checks': {'six_components': 'passed',
                                                                            'head_dim_72': 'passed',
                                                                            'full_runtime_owned_window': 'passed',
                                                                            'atol': 2e-05,
                                                                            'rtol': 2e-05},
                                                'cpp_window_input_owned': True},
                                   'native_features': {'shape': [64, 1024],
                                                       'max_abs_error': 6.937980651855469e-05,
                                                       'mean_abs_error': 1.4625606682555636e-06,
                                                       'cosine_similarity': 0.9999999999062021,
                                                       'ncnn_revision': 'b0f97f43f266617ff5e5096d440dbe3f9a4c35b8',
                                                       'runtime': 'MiniCPMVisionRuntime::encode',
                                                       'model_directory': 'minicpm-final-ready',
                                                       'feature_sha256': '981e1a14ae104dec8b09b0ab216a1525958cb347aa0a2fff67bbc6ecda14587e',
                                                       'resources': {'returncode': 0,
                                                                     'elapsed_seconds': 89.07279372700214,
                                                                     'max_rss_kib': 2305772,
                                                                     'user_cpu_seconds': 76.505573,
                                                                     'system_cpu_seconds': 36.802079,
                                                                     'language_model_loaded': False},
                                                       'runtime_source_sha256': '9413615a0625fa0037a046ced0b16fb645430fbf38c1e71f2e7e29ddf6847903'}},
 'minicpm_v47_vqa_validation': {'model_repository': 'openbmb/MiniCPM-V-4.7-1B',
                                'revision': 'ef3ab3bf0120a60e1a2d2b52d984a79d8cb7c630',
                                'checkpoint_sha256': '8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002',
                                'ncnn_revision': 'b0f97f43f266617ff5e5096d440dbe3f9a4c35b8',
                                'image': {'source_url': 'https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png',
                                          'original_size': [406, 528],
                                          'original_sha256': 'e384f22e7a10381084d3dafba477cf0fa59fa671612027e87e675de5ad149441',
                                          'test_fixture': 'refract-448.png',
                                          'test_fixture_sha256': '969dbe26ad4dd2d94daad99b6b7c172ea7adf9f56305ee416d39794b9fbd299a',
                                          'processing': 'Pillow ImageOps.pad RGB448x448 bicubic, '
                                                        'background240; both reference and ncnn receive the '
                                                        'same saved fixture'},
                                'prompt': 'Describe this image in one short sentence.',
                                'reference_text': 'A red pencil with a yellow tip rests in a glass of water, '
                                                  'partially submerged.',
                                'native_text': 'A red pencil with a yellow tip rests in a glass of water, '
                                               'partially submerged.',
                                'text_exact_match': True,
                                'native_prefill_token': 32,
                                'reference_prefill_token': 32,
                                'native_next_position': 40,
                                'reference_next_position': 40,
                                'prompt_token_count': 98,
                                'reference_generated_ids': [32,
                                                            2438,
                                                            44579,
                                                            440,
                                                            264,
                                                            13358,
                                                            11220,
                                                            51442,
                                                            303,
                                                            264,
                                                            8720,
                                                            314,
                                                            2919,
                                                            11,
                                                            24443,
                                                            81179,
                                                            13,
                                                            248046],
                                'native_all_generated_ids_recorded': False,
                                'native_dtype': 'float32',
                                'native_threads': 4,
                                'native_gdr': 'naive',
                                'native_max_new_tokens': 32,
                                'native_sampling': {'do_sample': False, 'repetition_penalty': 1},
                                'elapsed_seconds_including_load': 141.14642795499822,
                                'max_rss_kib': 6441280,
                                'returncode': 0,
                                'scope': 'Linux CPU single 448x448 RGB thumbnail,16x; video/detail '
                                         'slices/multiple images/quantization/GPU unverified',
                                'runtime_build': 'LocalAgent adapter build; MiniCPM uses untied Gemm output '
                                                 'head, not the adapted shared LM-head',
                                'original_upstream_build': {'runtime': 'unmodified upstream CMake LM-head '
                                                                       'sources with matching ncnn SDK, no '
                                                                       'LocalAgent source adaptation',
                                                            'native_text': 'A red pencil with a yellow tip '
                                                                           'rests in a glass of water, '
                                                                           'partially submerged.',
                                                            'text_exact_match': True,
                                                            'returncode': 0,
                                                            'elapsed_seconds_including_load': 80.46013123800003,
                                                            'max_rss_kib': 6441892}}}

HISTORICAL_RESULTS['consolidation_validation'] = {'date_utc': '2026-10-14',
 'runtime_source_commit': 'a14fe16f49ea6ad36767ade62bba9d6d3d1d8e60',
 'cpp_runtime_changed': False,
 'added_test_files_before': 34,
 'added_test_files_after': 1,
 'ctest_passed': 7,
 'language_unit_tests_passed': 5,
 'vision_unit_tests_passed': 15,
 'python_unit_tests_skipped': 0,
 'unsupported_configs_rejected_before_staging': 29,
 'resize_byte_exact_cases': 1290,
 'official_pil_preprocess_cases': 20,
 'official_pil_max_abs': 0.0,
 'hf_position_cases': 252,
 'prompt_cases': 27,
 'tokenizer_cases': 6549,
 'native_cached_decoder_max_abs': 4.76837158203125e-07,
 'cached_decoder_lengths': [8, 9, 12],
 'cached_decoder_initial_states': ['explicit', 'omitted'],
 'trained_vision': {'seconds': 20.531424303000676,
                    'max_abs': 6.937980651855469e-05,
                    'cosine': 0.9999999999062036,
                    'previous_native_bitwise_equal': True},
 'trained_decoder': {'seconds': 5.540083792991936,
                     'trained_decoder': True,
                     'full_vqa': False,
                     'shape': [1024, 8],
                     'max_abs': 0.0004215240478515625,
                     'mean_abs': 1.2515714690408686e-05,
                     'cosine': 0.9999999999891479,
                     'relative_l2': 4.6627328734838095e-06,
                     'atol': 0.0005,
                     'rtol': 0.0005,
                     'passed': True},
 'two_image_vqa': {'image_1_vqa': {'seconds': 32.44446859299205,
                                   'text': 'A red pencil with a yellow tip rests in a glass of water, '
                                           'partially submerged.',
                                   'exact_match': True,
                                   'independent_hf_reference': True},
                   'image_2_vqa': {'seconds': 24.081603307000478,
                                   'text': 'A bumblebee rests on a pink cosmos flower, surrounded by wilting '
                                           'blooms and red flowers in a garden.',
                                   'exact_match': True,
                                   'independent_hf_reference': False}},
 'converted_model_files_hash_unchanged': 25,
 'no_python_runtime_configuration_passed': True,
 'arbitrary_build_directory_generation_passed': True,
 'optimized_python_rejected': True}

# Public subcommands and lazily loaded groups. No generated Python files.
DRIVERS = {
    'resize': _resize_reference,
    'preprocess': _preprocess_reference,
    'positions': _vision_reference,
    'prompt': _prompt,
    'tokenizer': _tokenizer_reference,
    'prepare-decoder': _prepare_minicpm_decoder_probe,
    'compare-decoder': _compare_minicpm_decoder_probe,
    'compare-trained-decoder': _compare_minicpm_trained_decoder,
    'vision-reference': _check_minicpm_v47_vision_parity,
    'text-reference': _v47_text_reference,
}


def run_driver(name, arguments):
    positionals = {
        'prompt': ('model_dir', 'probe'),
        'prepare-decoder': ('fixture_dir', 'case_dir', 'probe'),
        'compare-decoder': ('case_dir',),
        'compare-trained-decoder': ('fixture_dir', 'native_output_dir'),
    }
    if name in positionals:
        parser = argparse.ArgumentParser(prog=f'{THIS_FILE.name} {name}')
        for argument in positionals[name]:
            parser.add_argument(argument, nargs='?' if name == 'prepare-decoder' and argument == 'probe' else None)
        parser.parse_args(arguments)
    old_argv = sys.argv
    sys.argv = [str(THIS_FILE) + " " + name, *map(str, arguments)]
    try:
        driver = DRIVERS[name]()
        if hasattr(driver, "main"):
            driver.main()
    finally:
        sys.argv = old_argv


def python_child(command, *arguments):
    subprocess.run([sys.executable, str(THIS_FILE), command, *map(str, arguments)], check=True)


def run_unit_tests(factories):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(factory()) for factory in factories)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f'Required suite failed or skipped cases: {result.skipped}')


def load_tests(loader, tests, pattern):
    """Support unittest discovery without importing tensor dependencies at import time."""
    class RequiredSuites(unittest.TestCase):
        def test_language_and_streamed_reference(self):
            python_child('language')

        def test_vision_export(self):
            python_child('vision-export')
    return loader.loadTestsFromTestCase(RequiredSuites)


def configure_build(path):
    build = Path(path).resolve()
    if not (build / 'CMakeCache.txt').is_file():
        raise FileNotFoundError(f'Configure and build the repository first: {build}')
    os.environ['NCNN_LLM_BUILD_DIR'] = str(build)
    probe = probe_path
    os.environ['NCNN_COMPONENT_PROBE'] = str(probe('minicpm_component_probe', 'executable'))
    os.environ['NCNN_VISION_PROBE'] = str(probe('minicpm_trained_vision_probe', 'executable'))
    return build


def native(name, *arguments, **kwargs):
    probe = probe_path(name, 'executable')
    return subprocess.run([str(probe), *map(str, arguments)], check=True, **kwargs)


def cached_decoder_regression(args):
    configure_build(args.build_dir)
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError('Choose a new cached-decoder output directory; preserve existing fixtures')
    python_child('prepare-decoder', args.fixture_dir, destination,
                 probe_path('minicpm_decoder_probe', 'executable'))
    # Reuse the same independent HF references with runtime-style omitted caches.
    case = json.loads((destination / 'case.json').read_text())
    case['initial_states'] = {}
    omitted = destination / 'case-omitted.json'
    omitted.write_text(json.dumps(case, indent=2))
    native('minicpm_decoder_probe', destination / 'decoder.ncnn.param', destination / 'decoder.ncnn.bin',
           omitted, destination / 'native-omitted')
    compare_cached_decoder(destination, 'native-omitted')
    print('PASS: native synthetic prefill/decode lengths 8/9/12 with explicit and omitted initial caches')


def real_regression(args):
    """Reuse explicit local reference fixtures; run memory-heavy stages serially."""
    import numpy as np
    import time
    build = configure_build(args.build_dir)
    model = args.converted_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'single-image CPU FP32 16x', 'stages': {}}

    def record(name, command):
        start = time.monotonic()
        result = command()
        report['stages'][name] = {'seconds': time.monotonic() - start, **result}
        (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({name: report['stages'][name]}), flush=True)

    def vision():
        native('minicpm_trained_vision_probe', model, args.image, out / 'features.f32', 2)
        actual = np.fromfile(out / 'features.f32', dtype='<f4').reshape(64, 1024).astype(np.float64)
        expected = np.load(args.vision_reference).astype(np.float64)
        assert expected.shape == actual.shape and np.isfinite(expected).all() and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-4)
        cosine = float(np.dot(actual.ravel(), expected.ravel()) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        assert cosine >= .999999, cosine
        values = {'max_abs': float(np.max(np.abs(actual - expected))), 'cosine': cosine}
        if args.previous_vision:
            previous = np.fromfile(args.previous_vision, dtype='<f4').reshape(64, 1024)
            np.testing.assert_array_equal(actual, previous)
            values['previous_native_bitwise_equal'] = True
        return values
    with ReadOnlyVisionPackage(model) as package:
        record('trained_vision', vision)
        report['converted_vision_sha256'] = package.before

    def decoder():
        params = json.loads((model / 'model.json').read_text())['params']
        destination = out / 'decoder'
        native('minicpm_decoder_probe', model / params['decoder_param'], model / params['decoder_bin'],
               args.decoder_fixture / 'case.json', destination)
        python_child('compare-trained-decoder', args.decoder_fixture, destination)
        return json.loads((destination / 'trained-parity.json').read_text())
    record('trained_decoder', decoder)

    reference = json.loads(args.vqa_reference.read_text())
    assert reference.get('text') and reference.get('generated_ids'), 'Expected official HF VQA text/token reference'
    for name, image, expected, official in (
        ('image_1_vqa', args.image, reference['text'], True),
        ('image_2_vqa', args.second_image, args.second_expected_text, False),
    ):
        def vqa():
            result = native('test_minicpm_runtime', model, image, args.prompt, 32,
                            text=True, capture_output=True)
            (out / (name + '.stdout')).write_text(result.stdout)
            (out / (name + '.stderr')).write_text(result.stderr)
            answer = result.stdout.strip().splitlines()[-1]
            assert answer == expected, (name, answer, expected)
            if official:
                first = reference['generated_ids'][0]
                next_position = len(reference['input_ids']) + reference['rope_deltas'][0][0]
                assert f'prefill_token={first} next_position={next_position}' in result.stderr, result.stderr
            return {'text': answer, 'exact_match': True, 'independent_hf_reference': official,
                    'reference': str(args.vqa_reference) if official else 'explicit pre-change native answer'}
        record(name, vqa)
    print('PASS: trained vision, decoder, and two sequential image-question regressions', flush=True)


def main():
    if not __debug__:
        raise RuntimeError('MiniCPM tests require assertions; do not use python -O or PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    emit = sub.add_parser('emit-native', help='CMake-only: write embedded C++ into a build directory')
    emit.add_argument('--build-dir', type=Path, required=True, help='CMake binary directory; sources go in its minicpm-test-sources child')
    sub.add_parser('history', help='Print retained validation measurements, provenance and limitations')
    sub.add_parser('language', help='Synthetic language export plus streamed-reference parity, no weights')
    vision = sub.add_parser('vision-export', help='All synthetic vision export tests; no skipped cases allowed')
    vision.add_argument('--build-dir', default=os.environ.get('NCNN_LLM_BUILD_DIR', str(ROOT / 'build')))
    for name in ('smoke', 'numerical'):
        command = sub.add_parser(name, help='Required lightweight native/synthetic tests' if name == 'smoke' else 'Official metadata/Pillow/tokenizer numerical parity, no weight loading')
        command.add_argument('--build-dir', default=os.environ.get('NCNN_LLM_BUILD_DIR', str(ROOT / 'build')))
        if name == 'numerical':
            command.add_argument('--model-dir', type=Path, required=True, help='Existing local HF checkpoint metadata/tokenizer directory')
    cached = sub.add_parser('cached-decoder', help='Native prefill/decode parity against independent HF, with explicit and omitted caches')
    cached.add_argument('--build-dir', required=True)
    cached.add_argument('--fixture-dir', type=Path, required=True, help='Existing tiny 64-wide decoder.ts and decoder.ncnn.param/bin')
    cached.add_argument('--output-dir', type=Path, required=True, help='New scratch/build directory for regenerated oracle tensors')
    real = sub.add_parser('real', help='Serial trained visual/decoder parity and two-image VQA against saved references')
    real.add_argument('--build-dir', required=True)
    for name, help_text in (
        ('converted-dir', 'Existing converted FP32 model package (no reconversion)'),
        ('image', '448x448 RGB PNG used in the saved official reference'),
        ('vision-reference', 'Saved official HF features.npy, shape 64x1024'),
        ('decoder-fixture', 'Directory with case.json, input tensors and independent HF target.f32'),
        ('vqa-reference', 'Official HF JSON: text, generated_ids, input_ids, rope_deltas'),
        ('second-image', 'Second 448x448 regression PNG'),
        ('output-dir', 'Build/scratch directory for results'),
    ):
        real.add_argument('--' + name, type=Path, required=True, help=help_text)
    real.add_argument('--second-expected-text', required=True, help='Exact pre-change native answer; not claimed as independent HF parity')
    real.add_argument('--previous-vision', type=Path, help='Optional pre-change native feature dump for bitwise comparison')
    real.add_argument('--prompt', default='Describe this image in one short sentence.')
    for name, module in DRIVERS.items():
        command = sub.add_parser(name, add_help=False, help='Reference driver: ' + name)
        command.add_argument('arguments', nargs=argparse.REMAINDER)
    # Driver flags belong to the preserved oracle's own parser.
    if len(sys.argv) > 1 and sys.argv[1] in DRIVERS:
        run_driver(sys.argv[1], sys.argv[2:])
        return
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    if args.command == 'emit-native':
        build_root = args.build_dir.resolve()
        destination = build_root / 'minicpm-test-sources'
        if build_root == ROOT or build_root == ROOT / 'tests' or ROOT / 'tests' in build_root.parents:
            raise ValueError('Generated probes must stay in a build directory, never the tests source directory')
        destination.mkdir(parents=True, exist_ok=True)
        for name, source in NATIVE_SOURCES.items():
            path = destination / (name + '.cpp')
            if not path.exists() or path.read_text() != source:
                path.write_text(source)
    elif args.command == 'history':
        print(json.dumps(HISTORICAL_RESULTS, indent=2))
    elif args.command == 'language':
        _streamed_reference().self_test()
        run_unit_tests([_v47_language_export])
    elif args.command == 'vision-export':
        configure_build(args.build_dir)
        run_unit_tests([_v47_vision_export])
    elif args.command == 'smoke':
        build = configure_build(args.build_dir)
        for name in ('test_minicpm_vision', 'test_vision_utils'):
            native(name)
        native('test_minicpm_runtime', '--check-limits')
        native('test_minicpm_tokenizer_config', build / 'tokenizer-config-fixtures')
        python_child('language')
        python_child('vision-export', '--build-dir', build)
    elif args.command == 'numerical':
        configure_build(args.build_dir)
        python_child('resize')
        python_child('preprocess', '--model-dir', args.model_dir)
        python_child('positions', '--model-dir', args.model_dir)
        python_child('prompt', args.model_dir, probe_path('minicpm_prompt_probe', 'executable'))
        python_child('tokenizer', '--model-dir', args.model_dir)
    elif args.command == 'cached-decoder':
        cached_decoder_regression(args)
    elif args.command == 'real':
        real_regression(args)


if __name__ == '__main__':
    main()
