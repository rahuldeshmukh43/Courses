# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: Main

clean:
	rm main

Main:   
	$(CC) $(CFLAGS) $(OMPFLAG) main.cpp -o main
