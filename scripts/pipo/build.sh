#!/bin/bash

if [ "$1" == "debug" ]; then
    echo "Building Debug version..."
    mkdir -p build-debug
    cd build-debug
    cmake .. -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=ON
    cmake --build . --target llama-simple llama-pipo-perf llama-pipo-override -j 8
    echo "Debug build complete. Executable is in build-debug/bin/"
elif [ "$1" == "release" ]; then
    echo "Building Release version..."
    mkdir -p build-release
    cd build-release
    cmake .. -DGGML_CUDA=ON
    cmake --build . --target llama-simple llama-pipo-perf llama-pipo-override -j 8
    echo "Release build complete. Executable is in build-release/bin/"
else
    echo "Usage: $0 [debug|release]"
    exit 1
fi
