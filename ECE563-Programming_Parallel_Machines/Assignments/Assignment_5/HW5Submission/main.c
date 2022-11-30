#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#define DEBUG
#undef DEBUG

#define MAXLEVEL 18
#define SLEEP_TIME 1
#define SEED 10

/*
 * This version does traversal and updates a global variable
 * I am adding a doWork() while traversing the tree to see better timing
 * I am also using the same tree for comparing the two traversals
 * I am setting the seeds using time so as to check if different traversals always return the same count
 */

struct node {
   int val;
   int level;
   int size;
   struct node* l;
   struct node* r;
} nodeT;

void doWork(int t) {
   usleep(t); //usleep takes time in microsecs
}

struct node* build_serial(int level) {
	if (level < MAXLEVEL) {
		struct node* p = (struct node*) malloc(sizeof(nodeT));
		p->val = rand( )%2; //random value 0 or 1
		p->level = level;
		p->size = pow(2, p->level);
		p->l = build_serial(level+1);
		p->r = build_serial(level+1);
		#ifdef DEBUG
		printf("build val:%d level:%d size:%d \n", p->val, p->level, p->size);
		#endif
		return p;
	} else {
	  return NULL;
	}
}

void traverse_serial(struct node* p, int* count_ptr) {
	if (p == NULL) return;
	#ifdef DEBUG
	printf("serial val:%d level:%d size:%d\n", p->val, p->level, p->size);
	#endif
	if ((float)p->val < 0.5){
	   *count_ptr += 1;
	}
	doWork(SLEEP_TIME);
	if (p->l == NULL) return;
	else traverse_serial(p->l, count_ptr);
	if (p->r == NULL) return;
	else traverse_serial(p->r, count_ptr);
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

struct node* build_parallel(int level, int num_threads) {
	if (level < MAXLEVEL) {
		struct node* p = (struct node*) malloc(sizeof(nodeT));
		p->val = rand( )%2; //random value 0 or 1
		p->level = level;
		p->size = pow(2, p->level);

		#pragma omp task
		p->l = build_serial(level+1);
		//#pragma omp taskwait

		#pragma omp task
		p->r = build_serial(level+1);
		//#pragma omp taskwait

		#ifdef DEBUG
		printf("build val:%d level:%d size:%d \n", p->val, p->level, p->size);
		#endif
		return p;
	} else {
	  return NULL;
	}
}

void traverse_parallel(struct node *p, int* count_ptr, int num_threads){
	if (p->l){
	#pragma omp task if (p->l->size < num_threads) untied
	traverse_parallel(p->l, count_ptr, num_threads);
	}
	if (p->r){
	#pragma omp task if (p->r->size < num_threads) untied
	traverse_parallel(p->r, count_ptr, num_threads);
	}
	#pragma omp taskwait
	doWork(SLEEP_TIME);
	#pragma omp critical
	{
	if ((float)p->val < 0.5)
		{
			*count_ptr += 1;
		}
	#ifdef DEBUG
	int tid = omp_get_thread_num();
	printf("parallel val:%d level:%d size:%d count:%d tid: %d\n", p->val, p->level, p->size, *count_ptr,tid);
	#endif
	}//omp critical
}

int main( ) {
	int count=0, num_threads=0;
	double start, end;
	struct node *h, *hp;

	//set seed
	#ifdef DEBUG
	srand(SEED));
	#endif
	#ifndef DEBUG
	srand(time(NULL));
	#endif

	//---------------- sequential part ---------------------
	start = omp_get_wtime();
	h = build_serial(0);
	end = omp_get_wtime();
	printf("Serial build time: %lf\n\n", end-start);

	start = omp_get_wtime();
	traverse_serial(h, &count);
	end = omp_get_wtime();
	printf("Count of nodes less than 0.5: %d\n", count);
	printf("Time for serial traversal: %lf\n\n", end - start);


	//---------------- parallel part ---------------------
	count = 0;
	num_threads = omp_thread_count();
	printf("Num threads: %d\n", num_threads);

	//parallel build
	start = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp single
		hp = build_parallel(0, num_threads);
	}
	end = omp_get_wtime();
	printf("Parallel build time: %lf\n", end-start);

	//parallel traverse
	start = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp single
		traverse_parallel(h, &count, num_threads);
	}
	end = omp_get_wtime();
	printf("Count of nodes less than 0.5: %d\n", count);
	printf("Time for parallel traversal: %lf\n", end - start);
}

