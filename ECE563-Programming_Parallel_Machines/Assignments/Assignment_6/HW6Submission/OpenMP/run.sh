#!/bin/bash

#make clean
#make all

{
for nt in 1 4 8 16
do
./mm $nt 
echo '--------------------'
done
}|tee out.txt
