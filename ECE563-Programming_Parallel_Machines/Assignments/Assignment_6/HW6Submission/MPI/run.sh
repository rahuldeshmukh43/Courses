#!/bin/bash

NumProcs=$1

make clean
#make debug
#mpirun -np 2 -gdb ./mm 2

make all
mpirun -np $1 ./mm $1
