cmake_minimum_required(VERSION 3.15)
project(ncnn_llm LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

if(WIN32)
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()
if(MSVC)
    add_compile_options(/utf-8)
endif()

option(NCNN_LLM_ENABLE_VULKAN "Build with ncnn Vulkan support" ON)

# OpenMP
find_package(OpenMP REQUIRED)

# NCNN Dependency
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(deps_ncnn)

# JSON Dependency
include(deps_json)

# NCNN Internal Layer Headers
set(NCNN_INTERNAL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src/layer
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src/layer/x86
)
if(TARGET ncnn)
    get_target_property(_ncnn_bin_dir ncnn BINARY_DIR)
    if(_ncnn_bin_dir)
        list(APPEND NCNN_INTERNAL_INCLUDE_DIRS ${_ncnn_bin_dir}/src)
    endif()
endif()

# Tokenizer library
add_library(ncnn_tokenizer STATIC
    src/utils/tokenizer/bpe_tokenizer.cpp
    src/utils/tokenizer/unigram_tokenizer.cpp
)
target_include_directories(ncnn_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/tokenizer
)
target_link_libraries(ncnn_tokenizer PUBLIC ncnn)

# ncnn_llm static library
set(NCNN_LLM_SOURCES
    src/ncnn_embedding.cpp
    src/ncnn_llm_asr.cpp
    src/ncnn_llm_gpt.cpp
    src/ncnn_llm_laya.cpp
    src/ncnn_llm_ocr.cpp
    src/ncnn_text_runtime.cpp
    src/sampling.cpp
    src/kernel/gdr.cpp
    src/kernel/shortconv.cpp
    src/utils/image_utils.cpp
    src/utils/prompt.cpp
    src/utils/rope_embed.cpp
    src/utils/vision_rope.cpp
    src/utils/perf_profiler.cpp
    src/kernel/lm_head.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x86|i[3-6]86" OR CMAKE_GENERATOR_PLATFORM MATCHES "x64|X64|x86|Win32" OR WIN32)
    list(APPEND NCNN_LLM_SOURCES
        src/kernel/x86/gdr_x86.cpp
        src/kernel/x86/shortconv_x86.cpp
        src/kernel/x86/lm_head_x86.cpp
    )
    if(MSVC)
        set_source_files_properties(src/kernel/x86/lm_head_x86.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else()
        set_source_files_properties(src/kernel/x86/lm_head_x86.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    endif()
endif()

add_library(ncnn_llm STATIC ${NCNN_LLM_SOURCES})

target_include_directories(ncnn_llm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/x86
    ${NCNN_INTERNAL_INCLUDE_DIRS}
)

target_link_libraries(ncnn_llm PUBLIC
    ncnn_tokenizer
    ncnn
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
)

if(WIN32)
    target_link_libraries(ncnn_llm PUBLIC user32 gdi32)
endif()

# Executables
# llm_ncnn_run
file(GLOB LLM_NCNN_RUN_SOURCES examples/llm_ncnn_run/*.cpp)
add_executable(llm_ncnn_run ${LLM_NCNN_RUN_SOURCES})
target_include_directories(llm_ncnn_run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/llm_ncnn_run
)
target_link_libraries(llm_ncnn_run PRIVATE ncnn_llm)
if(WIN32)
    target_link_libraries(llm_ncnn_run PRIVATE shell32)
endif()

# benchllm
add_executable(benchllm benchmark/benchllm.cpp)
target_include_directories(benchllm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(benchllm PRIVATE ncnn_llm)

# test_llm
add_executable(test_llm tests/test_llm.cpp)
target_include_directories(test_llm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_llm PRIVATE ncnn_llm)

# test_bf16
add_executable(test_bf16 tests/test_bf16.cpp)
target_include_directories(test_bf16 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_bf16 PRIVATE ncnn_llm)

# test_kernel
add_executable(test_kernel tests/test_kernel.cpp)
target_include_directories(test_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/x86
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${NCNN_INTERNAL_INCLUDE_DIRS}
)
target_link_libraries(test_kernel PRIVATE ncnn_llm)

# bench_qwen35
add_executable(bench_qwen35 tests/bench_qwen35.cpp)
target_include_directories(bench_qwen35 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
target_link_libraries(bench_qwen35 PRIVATE ncnn_llm)

# embedding_main
add_executable(embedding_main examples/embedding_main.cpp)
target_include_directories(embedding_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_link_libraries(embedding_main PRIVATE ncnn_llm)

# clip_main
add_executable(clip_main examples/clip_main.cpp)
target_include_directories(clip_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_link_libraries(clip_main PRIVATE ncnn_llm)

# ocr_main
add_executable(ocr_main examples/ocr_main.cpp)
target_include_directories(ocr_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_link_libraries(ocr_main PRIVATE ncnn_llm)
if(WIN32)
    target_link_libraries(ocr_main PRIVATE shell32)
endif()

# asr_main
add_executable(asr_main examples/asr_main.cpp)
target_include_directories(asr_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_link_libraries(asr_main PRIVATE ncnn_llm)
if(WIN32)
    target_link_libraries(asr_main PRIVATE shell32)
endif()

# laya_main
add_executable(laya_main examples/laya_main.cpp)
target_include_directories(laya_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_link_libraries(laya_main PRIVATE ncnn_llm)
if(WIN32)
    target_link_libraries(laya_main PRIVATE shell32)
endif()

# CTest
include(CTest)
enable_testing()
add_test(NAME test_kernel COMMAND test_kernel)
add_test(NAME test_llm COMMAND test_llm)
add_test(NAME test_bf16 COMMAND test_bf16)

# Install
install(TARGETS
    ncnn_llm
    ncnn_tokenizer
    llm_ncnn_run
    benchllm
    test_llm
    test_bf16
    test_kernel
    bench_qwen35
    embedding_main
    clip_main
    ocr_main
    asr_main
    laya_main
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# ncnn_llm_server: OpenAI-compatible HTTP service (third-party benchmarks / serving)
if(TARGET ncnn_llm)
    add_executable(ncnn_llm_server examples/ncnn_llm_server.cpp)
    target_include_directories(ncnn_llm_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    find_package(Threads REQUIRED)
    target_link_libraries(ncnn_llm_server PRIVATE ncnn_llm Threads::Threads)
endif()
