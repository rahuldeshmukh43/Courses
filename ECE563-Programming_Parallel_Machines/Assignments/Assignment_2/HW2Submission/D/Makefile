# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: verySlow slow

clean:
	rm verySlow slow

verySlow:   
	$(CC) $(CFLAGS) verySlow.c -o verySlow
	
slow:   
	$(CC) $(CFLAGS) slow.c -o slow
