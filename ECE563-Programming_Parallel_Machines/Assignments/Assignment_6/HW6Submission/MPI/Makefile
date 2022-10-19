# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=mpiicpc
CFLAGS= -g 

all: mm

clean:
	rm mm
mm:   
	$(CC) mm.c -o mm

debug:   
	$(CC) $(CFLAGS) mm.c -o mm
