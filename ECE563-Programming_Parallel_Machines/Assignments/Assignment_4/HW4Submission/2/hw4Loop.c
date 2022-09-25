#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
using namespace std;

#define DEBUG
#undef DEBUG

//int N=1000;

void doWork(int t) {
//   sleep(t); //sleep takes time in secs
   usleep(t); //usleep takes time in microsecs
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

int omp_thread_count() {
	/* for counting num threads in case of gcc */
    int n = 0;
    #pragma omp parallel
    {
		#pragma omp single
    	n = omp_get_num_threads();
    }
    return n;
}


int main (int argc, char *argv[]) {
	int i;
	double start, end;

	//parse args
	if(argc<3){
		printf("Provide size, sleep time and num threads as arg");
		return 1;
	}
	int Size = atoi(argv[1]);
	int sleep_time = atoi(argv[2]);
	int nthreads = atoi(argv[3]);

	int *w = initWork(Size);

	// get num threads
	omp_set_num_threads(nthreads);
	int Num_threads = omp_thread_count();
	printf("Num Threads: %d \n", Num_threads);
	printf("Array Size: %d\n", Size);
	printf("Sleep time in (usecs): %d\n\n", sleep_time);

//	// Static Scheduling with default block size
//	start = omp_get_wtime();
//	#pragma omp parallel for schedule(static)
//	for (i = 0; i < Size; i++) {
//		#ifndef DEBUG
//		doWork(sleep_time);
//		#endif
//		#ifdef DEBUG
//		printf("w[%d] = %d\n", i, w[i]);
//		#endif
//	}
//	end = omp_get_wtime();
//	printf("Time for static scheduling with default block size: %lf\n\n", end- start);

//	//Static+ BS 50
//	start = omp_get_wtime();
//	#pragma omp parallel for schedule(static, 50)
//	for (i = 0; i < Size; i++) {
//		#ifndef DEBUG
//		doWork(sleep_time);
//		#endif
//		#ifdef DEBUG
//		printf("w[%d] = %d\n", i, w[i]);
//		#endif
//	}
//	end = omp_get_wtime();
//	printf("Time for static scheduling with block size 50: %lf\n\n", end- start);

//	// dynamic default bs
//	start = omp_get_wtime();
//	#pragma omp parallel for schedule(dynamic)
//	for (i = 0; i < Size; i++) {
//		#ifndef DEBUG
//		doWork(sleep_time);
//		#endif
//		#ifdef DEBUG
//		printf("w[%d] = %d\n", i, w[i]);
//		#endif
//	}
//	end = omp_get_wtime();
//	printf("Time for dynamic scheduling with default block size: %lf\n\n", end- start);

//	// dynamic + bs 50
//	start = omp_get_wtime();
//	#pragma omp parallel for schedule(dynamic, 50)
//	for (i = 0; i < Size; i++) {
//		#ifndef DEBUG
//		doWork(sleep_time);
//		#endif
//		#ifdef DEBUG
//		printf("w[%d] = %d\n", i, w[i]);
//		#endif
//	}
//	end = omp_get_wtime();
//	printf("Time for dynamic scheduling with block size 50: %lf\n\n", end- start);

	// guided
	start = omp_get_wtime();
	#pragma omp parallel for schedule(guided)
	for (i = 0; i < Size; i++) {
		#ifndef DEBUG
		doWork(sleep_time);
		#endif
		#ifdef DEBUG
		printf("w[%d] = %d\n", i, w[i]);
		#endif
	}
	end = omp_get_wtime();
	printf("Time for guided scheduling with default block size: %lf\n\n", end- start);
}

