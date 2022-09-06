#!/bin/bash

make clean
make all

export OMP_NUM_THREADS=10
./A | tee A_out.txt


