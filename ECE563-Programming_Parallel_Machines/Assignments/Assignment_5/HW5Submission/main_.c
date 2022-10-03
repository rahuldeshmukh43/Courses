#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>

void doWork(int t) {
//   sleep(t); //sleep takes time in secs
   usleep(t); //usleep takes time in microsecs
}

struct node {
   int val;
   int level;
   int size;
   struct node* l;
   struct node* r;
} nodeT;

#define MAXLEVEL 18

struct node* build_serial(int level) {
	if (level < MAXLEVEL) {
		struct node* p = (struct node*) malloc(sizeof(nodeT));
		p->val = rand( )%2; //random value 0 or 1
		p->level = level;
		p->size = pow(2, p->level);
		p->l = build_serial(level+1);
		p->r = build_serial(level+1);
//		printf("build val:%d level:%d size:%d \n", p->val, p->level, p->size);
		return p;
	} else {
	  return NULL;
	}
}

void traverse_serial(struct node* p, int* count_ptr) {
	if (p == NULL) return;
//	printf("serial val:%d level:%d size:%d\n", p->val, p->level, p->size);
	if ((float)p->val < 0.5){
	   *count_ptr += 1;
	}
	doWork(1);
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
		#pragma omp task untied //if (p->size < num_threads)
		p->l = build_serial(level+1);
		#pragma omp task untied //if (p->size < num_threads)
		p->r = build_serial(level+1);
//		printf("build val:%d level:%d size:%d \n", p->val, p->level, p->size);
		return p;
	} else {
	  return NULL;
	}
}

//void traverse_parallel(struct node *p, int* count_ptr, int num_threads){
int traverse_parallel(struct node *p, int num_threads){
//	if(p->size > num_threads){
//		printf("size:%d too many nodes at level %d, going serial\n",p->size, p->level);
//		traverse_serial(p, count_ptr);
//	}r
	int cL=0, cR=0;
	if (p->l){
	#pragma omp task if (p->l->size < num_threads) shared(cL) untied
//	traverse_parallel(p->l, count_ptr, num_threads);
	cL = traverse_parallel(p->l, num_threads);
	}
	if (p->r){
	#pragma omp task if (p->r->size < num_threads) shared(cR) untied
//	traverse_parallel(p->r, count_ptr, num_threads);
	cR = traverse_parallel(p->r, num_threads);
	}
	#pragma omp taskwait
	doWork(1);
	if ((float)p->val < 0.5)
		{
			return cL+cR+1;
		}
	else{return cL+cR;}

//	#pragma omp critical
//	{
//	int tid = omp_get_thread_num();
//	if ((float)p->val < 0.5)
//		{
//			*count_ptr += 1;
//		}
////	printf("parallel val:%d level:%d size:%d count:%d tid: %d\n", p->val, p->level, p->size, *count_ptr,tid);
//	}//omp critical
}

int main( ) {
	int count=0, num_threads=0;
	double start, end;

   //sequential part
	start = omp_get_wtime();
//	struct node* h = build_serial(0);
	struct node* h;
	#pragma omp parallel //default(none)
	{
	#pragma omp single
	h = build_parallel(0, num_threads);
	}

	end = omp_get_wtime();
	printf("Serial build time: %lf\n\n", end-start);

	start = omp_get_wtime();
	traverse_serial(h, &count);
	end = omp_get_wtime();
	printf("Count of nodes less than 0.5: %d\n", count);
	printf("Time for serial traversal: %lf\n\n", end - start);


   //parallel part
	count = 0;
	num_threads = omp_thread_count();
	printf("Num threads: %d\n", num_threads);

	//parallel build
//	start = omp_get_wtime();
//	struct node* hp = build_parallel(0, num_threads);
//	end = omp_get_wtime();
//	printf("Parallel build time: %lf\n", end-start);
	//parallel traverse
	start = omp_get_wtime();
	#pragma omp parallel //default(none)
	{
		#pragma omp single //default(none) nowait
//		traverse_parallel(h, &count, num_threads);
		count = traverse_parallel(h, num_threads);
	}
	end = omp_get_wtime();
	printf("count of nodes less than 0.5: %d\n", count);
	printf("Time for parallel traversal: %lf\n", end - start);
}

