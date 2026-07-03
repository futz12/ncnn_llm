cmake_minimum_required(VERSION 3.16)
project(ncnn_llm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Include directories
include_directories(src)

# Find packages
find_package(ncnn REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)

# Static library: ncnn_tokenizer
add_library(ncnn_tokenizer STATIC
    src/utils/tokenizer/bpe_tokenizer.cpp
    src/utils/tokenizer/unigram_tokenizer.cpp
)

# Static library: ncnn_llm
file(GLOB NCNN_LLM_SOURCES
    src/*.cpp
    src/utils/*.cpp
)
add_library(ncnn_llm STATIC ${NCNN_LLM_SOURCES})
add_dependencies(ncnn_llm ncnn_tokenizer)
target_link_libraries(ncnn_llm PUBLIC
    ncnn_tokenizer
    ncnn
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
)

# Executable: llm_ncnn_run
file(GLOB LLM_NCNN_RUN_SOURCES examples/llm_ncnn_run/*.cpp)
add_executable(llm_ncnn_run ${LLM_NCNN_RUN_SOURCES})
target_include_directories(llm_ncnn_run PRIVATE examples)
target_link_libraries(llm_ncnn_run PRIVATE ncnn_llm)

# Executable: benchllm
add_executable(benchllm benchmark/benchllm.cpp)
target_link_libraries(benchllm PRIVATE ncnn_llm)

# Executable: test_llm
add_executable(test_llm tests/test_llm.cpp)
target_include_directories(test_llm PRIVATE tests)
target_link_libraries(test_llm PRIVATE ncnn_llm)

# Executable: nllb_main
add_executable(nllb_main examples/nllb_main.cpp)
target_link_libraries(nllb_main PRIVATE ncnn_llm)

# Executable: embedding_main
add_executable(embedding_main examples/embedding_main.cpp)
target_link_libraries(embedding_main PRIVATE ncnn_llm)

# Executable: clip_main
add_executable(clip_main examples/clip_main.cpp)
target_link_libraries(clip_main PRIVATE ncnn_llm)
