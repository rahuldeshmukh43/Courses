# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: C

clean:
	rm C

C:   
	$(CC) $(CFLAGS) $(OMPFLAG) C.cpp -o C