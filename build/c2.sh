# rm ./bin/cannonMM-cuda
# # cd ../deps-install/legion-build
# # cmake ../ -DCMAKE_BUILD_TYPE=Debug
# # make -j20 taco-test && bin/taco-test --gtest_filter="distributed.cuda_cannonMM" &&
# make -j20 cannonMM-cuda

rm ./bin/pummaMM-cuda
# cmake ../ -DCMAKE_BUILD_TYPE=Debug
# make -j20 taco-test && bin/taco-test --gtest_filter="distributed.cuda_pummaMM" &&
make -j20 pummaMM-cuda
