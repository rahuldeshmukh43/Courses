#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define DEBUG
#undef DEBUG

#define chunk_start(N, nthreads, tid) (tid*(N/nthreads))
#define chunk_end(N, nthreads, tid) ((tid+1)*(N/nthreads))

using namespace std;

void doWork(int t) {
//   sleep(t); //sleep takes time in secs
   usleep(t); //usleep takes time in microsecs
}

int* initWork(int n) {
   int i;
//   double r;
   int* wA = (int *) malloc(sizeof(int)*n);   
   for (i = 0; i < n; i++) {
      wA[i] = (int) rand( )%2*i/(n/10);
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
	if(argc<2){
		printf("Provide size of vector and sleep time as arg");
		return 1;
	}
	int Size = atoi(argv[1]);
	int sleep_time = atoi(argv[2]);

	int *w = initWork(Size);

	omp_set_num_threads(4);
	// get num threads
	int Num_threads = omp_thread_count();
	printf("Num Threads: %d \n", Num_threads);
	printf("Sleep time in (usecs): %d\n\n", sleep_time);

	//parallel section time (assume only 4 threads)
	printf("Parallel execution using sections\n");
	start = omp_get_wtime();
	#pragma omp parallel sections
	{
		//section 0
		#pragma omp section
		{
			int from=chunk_start(Size, Num_threads, 0 );
			int to=chunk_end(Size, Num_threads, 0 );
			printf("tid 0 start:%d stop:%d\n", from, to);
			for(int i=from; i<to; i++)
			{
			#ifdef DEBUG
			printf("section 0 working on w[%d]\n",i);
			#endif
			#ifndef DEBUG
			doWork(sleep_time);
			#endif
			}
		}

		//section 1
		#pragma omp section
		{
			int from=chunk_start(Size, Num_threads, 1);
			int to=chunk_end(Size, Num_threads, 1);
			printf("tid 1 start:%d stop:%d\n", from, to);
			for(i=from; i<to; i++)
			{
			#ifdef DEBUG
			printf("section 1 working on w[%d]\n",i);
			#endif
			#ifndef DEBUG
			doWork(sleep_time);
			#endif
			}
		}

		//section 2
		#pragma omp section
		{
			int from=chunk_start(Size, Num_threads, 2);
			int to=chunk_end(Size, Num_threads, 2);
			printf("tid 2 start:%d stop:%d\n", from, to);
			for(int i=from; i<to; i++)
			{
			#ifdef DEBUG
			printf("section 2 working on w[%d]\n",i);
			#endif
			#ifndef DEBUG
			doWork(sleep_time);
			#endif
			}
		}

		//section 3
		#pragma omp section
		{
			int from=chunk_start(Size, Num_threads, 3);
			int to=chunk_end(Size, Num_threads, 3);
			printf("tid 3 start:%d stop:%d\n", from, to);
			for(int i=from; i<to; i++)
			{
			#ifdef DEBUG
			printf("section 3 working on w[%d]\n",i);
			#endif
			#ifndef DEBUG
			doWork(sleep_time);
			#endif
			}
		}
	}
	end = omp_get_wtime();
	printf("Time for parallel (Array size %d): %lf\n\n", Size, end - start);

	//sequential execution time
	printf("Sequential Execution\n");
	start = omp_get_wtime();
	for(i=0; i<Size; i++){
		#ifdef DEBUG
		printf("sequential working on w[%d]\n",i);
		#endif
		#ifndef DEBUG
		doWork(sleep_time);
		#endif
	}
	end = omp_get_wtime();
	printf("Time for sequential (Array size %d): %lf\n\n",Size, end - start);
}

