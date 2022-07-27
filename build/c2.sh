rm ./bin/cannonMM-cuda
# cd ../deps-install/legion-build
cmake ../ -DCMAKE_BUILD_TYPE=Debug
# make -j20 taco-test && bin/taco-test --gtest_filter="distributed.cuda_cannonMM" && 
make -j20 cannonMM-cuda
