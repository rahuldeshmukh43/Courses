#!/bin/bash

make clean
make all

{
echo Slow 
./slow

echo ----------------
echo verySlow
./verySlow
} | tee D_out.txt
