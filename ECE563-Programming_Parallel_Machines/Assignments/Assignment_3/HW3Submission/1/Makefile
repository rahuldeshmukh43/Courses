# Makefile
#CC=icc
#OMPFLAG=-qopenmp

CC=g++
CFLAGS= -Wall -pedantic
OMPFLAG=-fopenmp


all: HW4A HW4B HW4C

clean:
	rm hw4a hw4b hw4c

HW4A:   
	$(CC) $(CFLAGS) $(OMPFLAG) hw4a.c -o hw4a
	
HW4B:   
	$(CC) $(CFLAGS) $(OMPFLAG) hw4b.c -o hw4b
	
HW4C:   
	$(CC) $(CFLAGS) $(OMPFLAG) hw4c.c -o hw4c