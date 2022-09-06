#/bin/bash
make clean
make all

export OMP_NUM_THREADS=10
./C | tee C_out.txt
