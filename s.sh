git clone git@github.com:Anjiang-Wei/taco
git checkout DISTAL
git submodule update --init deps/OpenBLAS deps/tblis deps/cub deps/legion

# for sapling
python3 scripts/install.py --openmp --sockets 2 --cuda --dim 4 --multi-node --threads 20 --tblis

# invokes the compiler to generate code for cannon's matmul algorithm
cd build
make -j taco-test && bin/taco-test --gtest_filter="distributed.cannonMM"

# cannons algorithm on a matrix of size 8192x8192 on a 2x1 processor grid of openmp procs
make -j cannonMM

# cannons algorithm on a matrix of size 8192x8192 on a 2x1 processor grid of openmp procs
./bin/cannonMM -n 8192 -gx 2 -gy 1 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0
