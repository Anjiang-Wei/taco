# Installation

Install TBLIS as follows:
```
./configure --prefix="$TBLIS_PREFIX" --enable-thread-model=openmp --enable-config=auto --without-blas
make -j4
make install

# prefix must be an ABSOLUTE path
# BLAS is only used for the benchmark program, and hence disabled.
```

Later when building TACO, make sure to specify `TBLIS_ROOT=$TBLIS_PREFIX` with the same prefix as used above.
