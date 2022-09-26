#!/bin/bash

#make clean
#make all

N=$1 
Usleep_time=$2

./hw4Q $N $Usleep_time | tee out_array_size_$N.txt
