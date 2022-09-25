#!/bin/bash

#make clean
#make all

ArraySize=$1 # default 1000
SleepTime=$2 # default 10 usecs
NumThreads=$3 #default 4 threads

./hw4Loop $ArraySize $SleepTime $NumThreads | tee out.txt
