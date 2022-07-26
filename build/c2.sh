# cd ../deps-install/legion-build
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DLegion_BOUNDS_CHECKS=ON -DLegion_PRIVILEGE_CHECKS=ON -DLegion_SPY=ON
make -j taco-test && bin/taco-test --gtest_filter="distributed.cuda_cannonMM" && make -j cannonMM-cuda
