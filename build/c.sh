rm bin/cannonMM
# cmake ../ -DCMAKE_BUILD_TYPE=Debug
# make -j20 taco-test && bin/taco-test --gtest_filter="distributed.cannonMM"
make -j20 cannonMM
