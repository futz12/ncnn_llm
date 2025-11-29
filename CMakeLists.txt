cmake_minimum_required(VERSION 3.20)
project(ncnn_llm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

if(MSVC)
  add_compile_options("/source-charset:utf-8")
  add_compile_definitions(NOMINMAX)
endif()

set(OpenCV_DIR "C:/pkgs/opencv/4.12.0/windows/x64/vc16/lib")
find_package(OpenCV REQUIRED)

#set(ncnn_DIR "C:/pkgs/ncnn/20250916/windows-vs2022-shared/x64/lib/cmake/ncnn")
set(ncnn_DIR "c:/pkgs/ncnn/master/vs2022-x64/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
include_directories("C:/pkgs/ncnn/master/vs2022-x64/include")

file(GLOB ncnn_tokenizer_src "src/utils/tokenizer/*.cpp")
add_library(ncnn_tokenizer STATIC ${ncnn_tokenizer_src})

file(GLOB ncnn_llm_src "src/*.cpp")
add_library(ncnn_llm STATIC ${ncnn_llm_src})
target_include_directories(ncnn_llm PUBLIC src)
target_link_libraries(ncnn_llm PUBLIC
  ncnn_tokenizer
  ncnn
  ${OpenCV_LIBS}
)

macro(add_example repo)
  add_executable(${repo} "${CMAKE_SOURCE_DIR}/examples/${repo}.cpp")
  target_include_directories(${repo} PRIVATE "examples")
  target_link_libraries(${repo} PRIVATE ncnn_llm ${OpenCV_LIBS})
  if(MSVC)
    set_target_properties(${repo} PROPERTIES
      VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
  endif()
endmacro()

add_example(nllb_main)
add_example(minicpm4_main)
add_example(bytelevelbpe_main)

add_executable(benchllm "benchmark/benchllm.cpp")
target_link_libraries(benchllm PRIVATE ncnn_llm)
