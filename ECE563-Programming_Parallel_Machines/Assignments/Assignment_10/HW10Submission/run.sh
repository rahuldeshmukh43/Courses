#!/bin/bash

make clean
make all

./dotproduct | tee dotproduct_out.txt
