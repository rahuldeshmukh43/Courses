# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: B

clean:
	rm B

B:   
	$(CC) $(CFLAGS) $(OMPFLAG) B.cpp -o B