# rm -rf build
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DNGP_BUILD_WITH_GUI=off -DCUDA_NVCC_FLAGS="-I$CONDA_PREFIX/include"
cmake --build build --config Release -j 16
echo "build completed!"
