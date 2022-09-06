#/bin/bash
make clean
make all

export OMP_NUM_THREADS=10
./B | tee B_out.txt
