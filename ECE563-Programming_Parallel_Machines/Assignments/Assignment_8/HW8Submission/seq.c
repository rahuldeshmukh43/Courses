#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
// #define ARRAY_ROWS 1600
// #define ARRAY_COLS 1600
#define ARRAY_ROWS 16
#define ARRAY_COLS 16

void printArray(int* a, int rows, int cols, int sparseness) {
   printf("rows: %d, cols: %d, sparseness: %d\n", rows, cols, sparseness);
   fflush(stdout);
   for (int i=0; i<rows; i+=sparseness) {
      for (int j=0; j<cols; j+=sparseness) {
         // printf("%.2f ", *(a + i*cols + j));
         printf("%d ", *(a + i*cols + j));
      }
      printf("\n");
   }
   printf("\n\n\n");
}

int* makeArray(int rows, int cols) {
   int* arr = (int*) malloc(rows*cols*sizeof(int));
   if (arr == NULL) printf("bad allocation"); fflush(stdout);

   for (int r=0; r<rows; r++) {
      for (int c=0; c<cols; c++) {
         *(arr + r*cols + c) = r+c;
      }
   }

   return arr;
}

int main (int argc, char *argv[]) {

   int* a = makeArray(ARRAY_ROWS, ARRAY_ROWS);
   int* b = makeArray(ARRAY_ROWS, ARRAY_ROWS);
   int* sendBuffer = makeArray(ARRAY_ROWS, ARRAY_ROWS);
   int* c = makeArray(ARRAY_ROWS, ARRAY_ROWS);

   for (int i=0; i<ARRAY_ROWS; i++) {
      for (int j=0; j<ARRAY_COLS; j++) {
         int comp = 0.;
         for (int k=0; k<ARRAY_ROWS; k++) {
            comp += *(a + i*ARRAY_COLS + k) * *(b + k*ARRAY_COLS + j);
         }
         *(c + i*ARRAY_COLS + j) = comp;
      }
   }

   printf("Array values:\n");
   printArray(c, ARRAY_ROWS, ARRAY_COLS, ARRAY_ROWS/16);
   return 0;
}
