#/bin/bash

make clean
make all

./main | tee out.txt

#./main_ | tee out_.txt
