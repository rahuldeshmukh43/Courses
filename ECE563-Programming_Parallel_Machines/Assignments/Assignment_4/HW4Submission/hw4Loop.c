#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void doWork(int t) {
   sleep(t);
}

int* initWork(int n) {
   int i;
   double r;
   int* wA = (int *) malloc(sizeof(int)*n);   
   for (i = 0; i < n; i++) {
      wA[i] = rand( )%2*i/(n/10);
   }
   return wA;
}

int main (int argc, char *argv[]) {
   int i;
   int *w = initWork(1000);
   for (i = 0; i < 1000; i+=50) {
      printf("w[%d] = %d\n", i, w[i]);
   }
}

