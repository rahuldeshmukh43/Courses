#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

#define DEBUG
#undef DEBUG

#define SLEEP 1

#define PROBLEMSIZE 10000000
#define NUMTHREADS 4

#define chunk_start(size, nthreads, tid) (tid*(size/nthreads))
#define chunk_end(size, nthreads, tid) ((tid+1)*(size/nthreads))


void doWork(int t) {
//   sleep(t); //sleep takes time in secs
   usleep(t); //usleep takes time in microsecs
}

double dproduct(float *a, float *b, int LB, int UB, int size){
	// sequential dot product
	double res=0;
	for(int i=0; i<size; i++){
		res += a[i]*b[i];
	#ifdef DEBUG
		doWork(SLEEP);
	#endif
	}
	return res;
}

double dproduct_tasks(float *a, float *b, int LB, int UB){
	// par dot prod using tasks
	double res=0;
	for(int i=LB; i<UB; i++){
		#ifdef DEBUG
			doWork(SLEEP);
		#endif
		res += a[i]*b[i];
	}
	return res;
}


double dproduct_omp_reduce(float *a, float *b, int size){
	// par dot prod using omp_reduce
	double res=0;
	#pragma omp parallel for reduction(+:res)
	for(int i=0; i<size; i++){
		#ifdef DEBUG
			doWork(SLEEP);
		#endif
		res += a[i]*b[i];
	}
	return res;
}

int main( ) {

   float* a;
   float* b;
   a = (float *) malloc(sizeof(float)*PROBLEMSIZE);
   b = (float *) malloc(sizeof(float)*PROBLEMSIZE);

   if (a==NULL) {
      printf("a is null\n");
      fflush(stdout);
   }

   if (b==NULL) {
      printf("b is null\n");
      fflush(stdout);
   }


   double res=0;
   double execTime=0;

   for (int i=0; i<PROBLEMSIZE; i++) {
      a[i] = b[i] = 1;
   }

   double *result = (double *) malloc(sizeof(double)*NUMTHREADS);
   if (result==NULL) {
 		printf("result is null\n");
 		fflush(stdout);
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

   omp_set_num_threads(NUMTHREADS);
   // parallel version with tasks
   execTime=0; res=0;
   execTime = -omp_get_wtime( );
   #pragma omp parallel
   {

		#pragma omp single
	   for(int i=0; i<NUMTHREADS; i++){
			#pragma omp task
		   result[i] = dproduct_tasks(a, b,
										chunk_start(PROBLEMSIZE, NUMTHREADS, i),
										chunk_end(PROBLEMSIZE, NUMTHREADS, i));
	   }
		#pragma omp taskwait
   }
   for(int i=0; i<NUMTHREADS; i++){
	res += result[i];
   }
   execTime += omp_get_wtime( );
   printf("dot product parallel result: %lf, time taken %lf\n", res, execTime);

   // OMP Reduction version
   execTime=0;
   execTime = -omp_get_wtime( );
   res = dproduct_omp_reduce(a, b, PROBLEMSIZE);
   execTime += omp_get_wtime( );
   printf("dot product omp reduction result: %lf, time taken %lf\n", res, execTime);

   return 0;
}



