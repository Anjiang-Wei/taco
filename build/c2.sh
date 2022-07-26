cmake ../ -DCMAKE_BUILD_TYPE=Debug -DLegion_BOUNDS_CHECKS=ON -DLegion_PRIVILEGE_CHECKS=ON -DLegion_SPY=ON
USE_SPY=1 make -j taco-test && USE_SPY=1 bin/taco-test --gtest_filter="distributed.cuda_cannonMM" && USE_SPY=1 make -j cannonMM-cuda
