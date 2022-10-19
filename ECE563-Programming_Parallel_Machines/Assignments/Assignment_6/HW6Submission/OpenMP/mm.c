#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

void printArray(double* a, int rows, int cols) {
   for (int i=0; i<rows; i++) {
      for (int j=0; j<cols; j++) {
         printf("%.2f ", *(a + i*cols + j));
      }
      printf("\n");
   }
   printf("\n\n\n");
}

double* makeArray(int rows, int cols) {
   double* arr = (double*) malloc(rows*cols*sizeof(double));

   for (int r=0; r<rows; r++) {
      for (int c=0; c<cols; c++) {
         *(arr + r*cols + c) = (double) (rows*c + c);
      }
   }

   return arr;
}

int min(int i, int j) {
   return i<j ? i : j;
} 

int main (int argc, char *argv[]) {

   const int ROWS = 1600;
   const int COLS = 1600;
   const int tasks = 1600;
   const int stripeSize = COLS/tasks;
   double start,end;

   //parse args
   if(argc<1){
	   printf("Provide number of threads as arg\n");
   }
   int NUM_THREADS = atoi(argv[1]);
   omp_set_num_threads(NUM_THREADS);

   double* a = makeArray(ROWS, COLS);
   double* b = makeArray(ROWS, COLS);
   double* c = makeArray(ROWS, COLS);

   start = omp_get_wtime( );
   #pragma omp parallel
   {
   	   #pragma omp single
	   {
	   for (int t=0; t<tasks; t++) {
		  #pragma omp task
		  {
			  for (int i=t*stripeSize; i<min(t*stripeSize+stripeSize, ROWS); i++) {
				 for (int j=0; j<COLS; j++) {
					double comp = 0.;
					for (int k=0; k<COLS; k++) {
					   comp += *(a + i*COLS + k) * *(b + k*COLS + j);
					}
					*(c + i*COLS + j) = comp;
				 }
			  }
		  }
	   }
	   }
   }
   end = omp_get_wtime();
//   double timeTaken = (timer + clock( ))/CLOCKS_PER_SEC;
   printf("time taken for matrix multiply (%d threads): %.2lf\n", NUM_THREADS, end-start);
   
   // printArray(c, ROWS, COLS);
}
