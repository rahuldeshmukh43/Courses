# Makefile
MPICC=mpiicpc
GCC=g++

CFLAGS= -Wall -pedantic
DEBUG_FLAG= -g 

all: mm seq

clean:
	rm mm seq
mm:   
	$(MPICC) mm.c -o mm
	
seq:
	$(GCC) $(CFLAGS) -O3 seq.c -o seq 
	
debug:   
	$(MPICC) $(DEBUG_FLAG) mm.c -o mm
