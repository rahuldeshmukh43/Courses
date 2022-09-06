#include <time.h>
#include <stdlib.h>
#include <cstdio>

#define NUMROWS 2520
#define NUMCOLS 2520
#define idx(u, y, x) (u[y*NUMCOLS + x])

float* newArray(int rows, int cols) {
   float* a = (float*) malloc(NUMROWS * NUMCOLS * sizeof(float)); 
   for (int i = 0; i < cols; i++) {
      for (int j = 0; j < cols; j++) {
         idx(a,i,j) = 1.0;
      }
   }
   return a;
}

int main(int argc, char** args) {

   float* a = newArray(NUMROWS, NUMCOLS);
   float* b = newArray(NUMROWS, NUMCOLS);
   float* c = newArray(NUMROWS, NUMCOLS);

   clock_t begin = clock();
   for (int i = 0; i < NUMROWS; i++) {
      for (int j = 0; j < NUMCOLS; j++) {
         float comp = 0.;
         for (int k = 0; k < NUMCOLS; k++) {
            comp += idx(a,i,k)*idx(b,k,j);
         }
         idx(c,i,j) = comp;
      }
   }
   clock_t end = clock();
   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("Elapsed: %f seconds\n", time_spent);
}
