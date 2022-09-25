#!/bin/bash

make clean
make all

N=$1 

./hw4Q $N | tee out_array_size_$N.txt
