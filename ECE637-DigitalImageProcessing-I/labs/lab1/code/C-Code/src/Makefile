# For Linux or any machines with gcc compiler
CC = gcc 
CFLAGS = -std=c99 -Wall -pedantic
BIN = ../bin

all: ImageReadWriteExample ImageReadWrite_FIR ImageReadWrite_FIR_sharpen ImageReadWrite_IIR SurrogateFunctionExample 

clean:
	/bin/rm *.o $(BIN)/*

OBJ = tiff.o allocate.o randlib.o qGGMRF.o solve.o

ImageReadWriteExample: ImageReadWriteExample.o $(OBJ) 
	$(CC) $(CFLAGS) -o ImageReadWriteExample ImageReadWriteExample.o $(OBJ) -lm
	mv ImageReadWriteExample $(BIN)

ImageReadWrite_FIR: ImageReadWrite_FIR.o $(OBJ) 
	$(CC) $(CFLAGS) -o ImageReadWrite_FIR ImageReadWrite_FIR.o $(OBJ) -lm
	mv ImageReadWrite_FIR $(BIN)

ImageReadWrite_FIR_sharpen: ImageReadWrite_FIR_sharpen.o $(OBJ) 
	$(CC) $(CFLAGS) -o ImageReadWrite_FIR_sharpen ImageReadWrite_FIR_sharpen.o $(OBJ) -lm
	mv ImageReadWrite_FIR_sharpen $(BIN)

ImageReadWrite_IIR: ImageReadWrite_IIR.o $(OBJ) 
	$(CC) $(CFLAGS) -o ImageReadWrite_IIR ImageReadWrite_IIR.o $(OBJ) -lm
	mv ImageReadWrite_IIR $(BIN)


SurrogateFunctionExample: SurrogateFunctionExample.o $(OBJ) 
	$(CC) $(CFLAGS) -o SurrogateFunctionExample SurrogateFunctionExample.o $(OBJ) -lm
	mv SurrogateFunctionExample $(BIN)
