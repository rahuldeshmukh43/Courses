#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define MEMSIZE 100000
#define NUMTHREADS 4

typedef struct node {
   int valIdx;
   int left;
   int right;
} nodeT;

int memIdx = 0;
nodeT* memory;
int data[10];

void initData( ) {
   for (int i=0; i<10; i++) {
      data[i] = 0;
   }
}

void init( ) {
   memIdx = 0;
   memory = (nodeT*) malloc(sizeof(nodeT)*MEMSIZE); 
   if (memory == NULL) {
      printf("invalid memory allocation\n");
      fflush(stdout);
   }
   initData( );
}

int myMalloc( ) {
   if (memIdx < MEMSIZE) {
      return memIdx++;
   }
   return -1;
}

int build(int count) {
   int me;
   if ((me = myMalloc( )) < 0) {
      return -1;
   }

   count = ++count % 10;
   memory[me].valIdx = count;
   memory[me].left = build(count);
   memory[me].right = build(count);
   return me;
}

int main( ) {

   init( );
   build(-1);
}
