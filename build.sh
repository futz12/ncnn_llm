#!/usr/bin/env bash

set -e  # 出错立即退出
set -u  # 未定义变量时报错

# === 1. 拉取submodule ===
echo "submodule update ..."
git submodule update --init --recursive

echo "start build ..."

# === 2. 创建并进入 build 目录 ===
mkdir -p build
pushd build

BUILD_TYPE=${BUILD_TYPE:-Release}
INSTALL_DIR=${INSTALL_DIR:-$(pwd)/install}
TOOLCHAIN_FILE=${TOOLCHAIN_FILE:-""}

echo "========== Build Config =========="
echo "BUILD_TYPE    = ${BUILD_TYPE}"
echo "INSTALL_DIR   = ${INSTALL_DIR}"
echo "TOOLCHAIN_FILE= ${TOOLCHAIN_FILE}"
echo "=================================="

# === 3. 编译NCNN   ===
echo "build ncnn ..."
mkdir -p ncnn
pushd ncnn
cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    ../../3rdparty/ncnn -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
     -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j4
make install
popd

# === 4. 编译OPENCV  ===
echo "build opencv ..." 
mkdir -p opencv
pushd opencv
cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    ../../3rdparty/opencv -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_OPENCV_PYTHON3=OFF -DBUILD_OPENCV_JAVA=OFF \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF
make -j4
make install
popd    

# === 5. 编译ncnn_llm ===
echo "build ncnn_llm ..."
mkdir -p ncnn_llm   
pushd ncnn_llm
cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -Dncnn_DIR=${INSTALL_DIR}/lib/cmake/ncnn \
    -DOpenCV_DIR=${INSTALL_DIR}/lib/cmake/opencv4 ../.. 
make -j4
make install
popd    