#!/bin/bash

make clean
make all

DIM=$1
./main $DIM | tee out.txt
