# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: Q1 Q2 Q3

clean:
	rm Q1 Q2 Q3

Q1:   
	$(CC) $(CFLAGS) $(OMPFLAG) Q1.c -o Q1

Q2:   
	$(CC) $(CFLAGS) $(OMPFLAG) Q2.c -o Q2

Q3:   
	$(CC) $(CFLAGS) $(OMPFLAG) Q3.cpp -o Q3

