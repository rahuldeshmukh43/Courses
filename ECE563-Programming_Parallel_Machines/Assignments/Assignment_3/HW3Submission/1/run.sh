#!/bin/bash

make clean

make all

echo hw4a
./hw4a | tee hw4a.txt

echo ------------------------
echo hw4b
./hw4b | tee hw4b.txt

echo ------------------------
echo hw4c
./hw4c | tee hw4c.txt