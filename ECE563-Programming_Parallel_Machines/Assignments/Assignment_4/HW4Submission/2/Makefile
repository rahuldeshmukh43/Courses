# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: hw4Loop

clean:
	rm hw4Loop

hw4Loop:   
	$(CC) $(CFLAGS) $(OMPFLAG) hw4Loop.c -o hw4Loop
