#!/bin/bash

make clean
make all

echo '------------- Q1 -----------------'
./Q1 | tee Q1.out

echo '------------- Q2 -----------------'
./Q2 | tee Q2.out

echo '------------- Q3 -----------------'
./Q3 | tee Q3.out

python Q3_plot.py serial.csv
python Q3_plot.py parallel.csv

