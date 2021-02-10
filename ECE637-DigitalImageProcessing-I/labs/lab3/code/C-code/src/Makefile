# For Linux or any machines with gcc compiler
CC = g++
#CFLAGS = -std=c++03 -Wall -pedantic
CFLAGS = -Wall -pedantic
BIN = ../bin

all: connected_set 

clean:
	/bin/rm *.o $(BIN)/*

OBJ = tiff.o allocate.o randlib.o qGGMRF.o solve.o

connected_set: ConnectedSet.o $(OBJ) 
	$(CC) $(CFLAGS) -o ConnectedSet ConnectedSet.o $(OBJ) -lm
	mv ConnectedSet $(BIN)
