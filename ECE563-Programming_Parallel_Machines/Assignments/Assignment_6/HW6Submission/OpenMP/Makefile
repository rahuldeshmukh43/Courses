# Makefile
#CC=icc
#OMPFLAG=-openmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: mm

clean:
	rm mm

mm:   
	$(CC) $(CFLAGS) $(OMPFLAG) mm.c -o mm
