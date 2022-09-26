#!/bin/bash

#make clean
#make all

ArraySize=$1
Usleep_time=$2 # in microsecs
./hw4Section $ArraySize $Usleep_time | tee out_size_$ArraySize.txt
