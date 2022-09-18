#!/bin/bash

make clean
make all

./main | tee out.txt
