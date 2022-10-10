#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define PROBLEMSIZE 10000000

int main( ) {

   float* a;
   float* b;
   a = malloc(sizeof(float)*PROBLEMSIZE);
   b = malloc(sizeof(float)*PROBLEMSIZE);

   if (a==NULL) {
      printf("a is null\n");
      fflush(stdout);
   }

   if (b==NULL) {
      printf("b is null\n");
      fflush(stdout);
   }


   double res=0;
   double execTime;

   for (int i=0; i<PROBLEMSIZE; i++) {
      a[i] = b[i] = 1;
   }

   for (int i=0; i<NUMTHREADS; i++) {
      result[i] = 0;
   }

   // sequential execution to check the answer
   execTime = -omp_get_wtime( );
   res = dproduct(a, b, 0, /* lower bound in the array of where to start the computation */
                        0, /* upper bound in the array of the last element+1 to process */
                  PROBLEMSIZE);
   execTime += omp_get_wtime( );
   printf("dot product sequential result: %lf, time taken %lf\n", res, execTime); fflush(stdout);


   // parallel version with tasks
   printf("dot product parallel result: %lf, time taken %lf\n", res, execTime);

   // OMP Reduction version
   printf("dot product omp reduction result: %lf, time taken %lf\n", res, execTime);
}
