# rm ./bin/cannonMM-cuda
# # cd ../deps-install/legion-build
# # cmake ../ -DCMAKE_BUILD_TYPE=Debug
# # make -j20 taco-test && bin/taco-test --gtest_filter="distributed.cuda_cannonMM" &&
# make -j20 cannonMM-cuda

# rm ./bin/pummaMM-cuda
# # cmake ../ -DCMAKE_BUILD_TYPE=Debug
# # make -j20 taco-test && bin/taco-test --gtest_filter="distributed.cuda_pummaMM" &&
# make -j20 pummaMM-cuda

# rm ./bin/summaMM-cuda
# # cmake ../ -DCMAKE_BUILD_TYPE=Release
# # make -j20 taco-test && bin/taco-test --gtest_filter="distributed.cuda_summaMM" &&
# make -j20 summaMM-cuda

rm ./bin/solomonikMM-cuda
# cmake ../ -DCMAKE_BUILD_TYPE=Debug
# make -j20 taco-test && bin/taco-test --gtest_filter="distributed.cuda_solomonikMM" &&
make -j20 solomonikMM-cuda
