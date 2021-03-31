# For Linux or any machines with gcc compiler
#CC = gcc 
CC=g++
#CFLAGS = -std=c99 -Wall -pedantic
CFLAGS= -Wall -pedantic
BIN = ../bin

all: Median_Filtering #ImageReadWriteExample SurrogateFunctionExample 

clean:
	/bin/rm *.o $(BIN)/*

OBJ = tiff.o allocate.o randlib.o qGGMRF.o solve.o

Median_Filtering: MedianFiltering.o $(OBJ)
	$(CC) $(CFLAGS) -o MedianFiltering MedianFiltering.o $(OBJ) -lm
	mv MedianFiltering $(BIN)

ImageReadWriteExample: ImageReadWriteExample.o $(OBJ) 
	$(CC) $(CFLAGS) -o ImageReadWriteExample ImageReadWriteExample.o $(OBJ) -lm
	mv ImageReadWriteExample $(BIN)

SurrogateFunctionExample: SurrogateFunctionExample.o $(OBJ) 
	$(CC) $(CFLAGS) -o SurrogateFunctionExample SurrogateFunctionExample.o $(OBJ) -lm
	mv SurrogateFunctionExample $(BIN)
