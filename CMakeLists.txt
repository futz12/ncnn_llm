cmake_minimum_required(VERSION 3.22)
project(ncnn_llm)

set(CMAKE_CXX_STANDARD 20)

find_package(ncnn REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB TOKENIZER_SRC "src/utils/tokenizer/*.cpp")

include_directories(src)

add_library(
        ncnn_tokenizer
        ${TOKENIZER_SRC}
)

add_library(
        ncnn_llm
        src/nllb_600m.cpp
)

target_link_libraries(ncnn_llm
        ncnn_tokenizer
        ncnn
#${OpenCV_LIBS}
)

# example
add_executable(
        unigram_main
        examples/unigram_main.cpp
)
target_include_directories(unigram_main PUBLIC src)
target_link_libraries(unigram_main ncnn_tokenizer)

add_executable(
        nllb_main
        examples/nllb_main.cpp
)
target_include_directories(nllb_main PUBLIC src)
target_link_libraries(nllb_main
        ncnn_llm
        ${OpenCV_LIBS}
)