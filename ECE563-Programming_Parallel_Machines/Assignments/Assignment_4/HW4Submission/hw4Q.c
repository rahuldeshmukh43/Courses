#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int N=100;

typedef struct Q {
   int* q;
   int pos;
   int size;
} Q;

struct Q* initQ(int n) {
   int i;
   struct Q *newQ = (struct Q *) malloc(sizeof(Q));   
   newQ->q = (int*) malloc(sizeof(int)*n);
   newQ->pos = -1;
   newQ->size = n-1;
   return Q;
}

void putWork(struct Q* workQ) {
   if (workQ->pos < (workQ->size)) {
      workQ->pos++;
      workQ->q[workQ->pos] = (int) (rand( )%2*(workQ->pos/1000));
   } else printf("ERROR: attempt to add Q element%d\n", workQ->pos+1);
}

int getWork(struct Q* workQ) {
   if (workQ->pos > -1) {
      int w = workQ->q[workQ->pos];
      workQ->pos--;
      return w;
   } else printf("ERROR: attempt to get work from empty Q%d\n", workQ->pos);
}

int main(){


	int w;
	double start, end;
	// initialize and add work to the work queue
	struct Q* Queue = initQ(N);
	// add work
	for(int i=0; i<Queue->size; i++){
		putWork(Queue);
	}


	// Loop within parallel construct that pulls work from work queue
	start = omp_get_wtime();
	#pragma omp parallel for
	for(i=0; i< Queue->size; i++){
		w = getWork(workQ);
	}
	end = omp_get_wtime();
	printf("Time to initialize: %lf\n\n", end - start);


	//put work again
	for(int i=0; i<Queue->size; i++){
		putWork(Queue);
	}


	// sequential
	start = omp_get_wtime();
	for(i=0; i< Queue->size; i++){
		w = getWork(workQ);
	}
	end = omp_get_wtime();
	printf("Time to initialize: %lf\n\n", end - start);

	return 0;
}
