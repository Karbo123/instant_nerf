# rm -rf build
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_NVCC_FLAGS="-I$CONDA_PREFIX/include"
cmake --build build --config Release -j 16 && echo "build completed!"
