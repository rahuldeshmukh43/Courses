#define DEBUG
#undef DEBUG
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

//int N=100;

typedef struct Q {
   int* q;
   int pos;
   int size;
} Q;

struct Q* initQ(int n) {
   struct Q *newQ = (struct Q *) malloc(sizeof(Q));   
   newQ->q = (int*) malloc(sizeof(int)*n);
   newQ->pos = -1;
   newQ->size = n;
   return newQ;
}

void putWork(struct Q* workQ) {
   if (workQ->pos < (workQ->size)) {
      workQ->pos++;
      workQ->q[workQ->pos] = (int) (rand( )%workQ->size*(workQ->pos));
   } else printf("ERROR: attempt to add Q element%d\n", workQ->pos+1);
}

int getWork(struct Q* workQ) {
   if (workQ->pos > -1) {
      int w = workQ->q[workQ->pos];
      workQ->pos--;
      return w;
   } else printf("ERROR: attempt to get work from empty Q%d\n", workQ->pos);
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


int main(int argc, char *argv[]){
	int w,i;
	double start, end;

	//parse args
	if(argc<1){
		printf("Provide size of vector as arg");
		return 1;
	}
	int N = atoi(argv[1]);

	// initialize and add work to the work queue
	struct Q* Queue = initQ(N);

	// get num threads
	int Num_threads = omp_thread_count();
	printf("Num Threads: %d \n\n", Num_threads);

	// add work
	printf("Putting work\n");
	for(int i=0; i<Queue->size; i++){
		putWork(Queue);
		#ifdef DEBUG
		printf("work: %d, pos: %d\n",Queue->q[Queue->pos], Queue->pos);
		#endif
	}
	printf("\n");


	// Loop within parallel construct that pulls work from work queue
	printf("Getting work parallel\n");
	start = omp_get_wtime();
	#pragma omp parallel for private(w)
	for(int j=0; j<Queue->size; j++){
		//#pragma omp critical
		w = getWork(Queue);
		#ifdef DEBUG
		printf("work: %d, pos: %d\n",w, Queue->pos);
		#endif
	}
	end = omp_get_wtime();
	printf("Time for parallel (Queue size %d): %lf\n\n", N, end - start);


	//put work again
	printf("Reset queue pos for sequential\n");
	Queue->pos=N-1;
//	for(i=0; i<Queue->size; i++){
//		putWork(Queue);
//		printf("work: %d, pos: %d\n",Queue->q[Queue->pos], Queue->pos);
//	}
//	printf("\n");

	// sequential
	printf("Getting work sequential\n");
	start = omp_get_wtime();
	for(i=0; i<Queue->size; i++){
		w = getWork(Queue);
		#ifdef DEBUG
		printf("work: %d, pos: %d\n",w, Queue->pos);
		#endif
	}
	end = omp_get_wtime();
	printf("Time for sequential (Queue size %d): %lf\n\n", N, end - start);

	return 0;
}


