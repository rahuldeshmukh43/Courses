# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: main main_

clean:
	rm main main_ 

main:   
	$(CC) $(CFLAGS) $(OMPFLAG) main.c -o main

main_:   
	$(CC) $(CFLAGS) $(OMPFLAG) main_.c -o main_
