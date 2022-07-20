cmake ../ -DCMAKE_BUILD_TYPE=Debug ../
make -j taco-test && bin/taco-test --gtest_filter="distributed.cuda_cannonMM" && make -j cannonMM-cuda
