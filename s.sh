# on sapling
git clone git@github.com:Anjiang-Wei/taco
git checkout DISTAL
git submodule update --init deps/OpenBLAS deps/tblis deps/cub deps/legion
python3 scripts/install.py --openmp --sockets 2 --cuda --dim 4 --multi-node --threads 20 --tblis
