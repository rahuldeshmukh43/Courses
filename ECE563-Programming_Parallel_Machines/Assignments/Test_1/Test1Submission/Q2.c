#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

#define MEMSIZE 100000
#define NUMTHREADS 4

//#define DEBUG
//#undef DEBUG
#define SLEEP 1

void doWork(int t) {
//   sleep(t); //sleep takes time in secs
   usleep(t); //usleep takes time in microsecs
}

typedef struct node {
   int valIdx;
   int left;
   int right;
} nodeT;

int memIdx = 0;
nodeT* memory;
int data[10];
int par_data[10];

void initData( int *data ) {
   for (int i=0; i<10; i++) {
      data[i] = 0;
   }
}

void init( ) {
   memIdx = 0;
   memory = (nodeT*) malloc(sizeof(nodeT)*MEMSIZE); 
   if (memory == NULL) {
      printf("invalid memory allocation\n");
      fflush(stdout);
   }
   initData( data );
   initData( par_data );
}

int myMalloc( ) {
   if (memIdx < MEMSIZE) {
      return memIdx++;
   }
   return -1;
}

int build(int count) {
	// this always builds a left branch only tree
   int me;
   if ((me = myMalloc( )) < 0) {
      return -1;
   }

   count = ++count % 10;
   memory[me].valIdx = count;
   memory[me].left = build(count);
   memory[me].right = build(count);
   return me;
}

void traverse_sequential(int idx, int *data ){
	#ifdef DEBUG
	doWork(SLEEP);
	#endif
	data[memory[idx].valIdx] += 1;
	if (memory[idx].left == -1) return;
	else traverse_sequential(memory[idx].left, data);
	if (memory[idx].right == -1) return;
	else traverse_sequential(memory[idx].right, data);
}

//void _traverse_parallel(int idx, int *data){
//	// parallel traversal assuming a generic binary tree
//	#pragma omp critical
//	{
//	#ifdef DEBUG
//	doWork(SLEEP);
//	#endif
//	data[memory[idx].valIdx] += 1;
//	}
//
//	if (memory[idx].left == -1) return;
//	else{
//		#pragma omp task untied
//		_traverse_parallel(memory[idx].left, data);
//	}
//
//	if (memory[idx].right == -1) return;
//	else{
//		#pragma omp task untied
//		{
//		#ifdef DEBUG
//		printf("go right\n");
//		#endif
//		_traverse_parallel(memory[idx].right, data);
//		}
//	}
//
//	#pragma omp taskwait
//}


//-------------------------------------------------------------------
//         Parallel Traversal using Array representation
//-------------------------------------------------------------------
//#define chunk_start(size, nthreads, tid) (tid*(size/nthreads))
//#define chunk_end(size, nthreads, tid) ((tid+1)*(size/nthreads))
//
//void traverse_array(int LB, int UB, int *data){
//	for(int i=LB; i<UB; i++){
//		#pragma omp critical
//		{
//			#ifdef DEBUG
//			doWork(SLEEP);
//			#endif
//			data[memory[i].valIdx] += 1;
//		}
//	}
//	return;
//}
//
//void traverse_parallel(int *data){
//	// parallel traversal assuming a the tree is always left (ie a arrray or LL)
//	for(int tid=0; tid<NUMTHREADS; tid++){
//		#pragma omp task untied
//		traverse_array(chunk_start(MEMSIZE, NUMTHREADS, tid),
//					   chunk_end(MEMSIZE, NUMTHREADS, tid),
//					   data);
//	}
//	#pragma omp taskwait
//	return;
//}

void traverse_parallel(int *data){
	int tmp;
	omp_lock_t lck[10];
	for(int i=0; i<10; i++){
		omp_init_lock(&(lck[i]));
	}

	#pragma omp parallel for private(tmp)
	for(int i=0; i<MEMSIZE; i++){
		int tmp = (memory[i].valIdx % 10);
		omp_set_lock(&lck[tmp]);
		#ifdef DEBUG
		doWork(SLEEP);
		#endif
		data[memory[i].valIdx] += 1;
		omp_unset_lock(&lck[tmp]);
	}

	for(int i=0; i<10; i++){
		omp_destroy_lock(&(lck[i]));
	}

}
//-------------------------------------------------------------------

int main( ) {
   double start, end;

   init( );
   build(-1);

   //sequential traversal
   start = omp_get_wtime();
   traverse_sequential(0, data);
   int sum=0;
   for (int i=0; i<10; i++){
	   sum += data[i];
	   printf("data[%d] %d\n", i , data[i]);
   }
   end = omp_get_wtime();
   printf("sum serial is %d, ", sum);
   printf("time is %lf\n\n", end - start);

   //parallel traversal
   omp_set_num_threads(NUMTHREADS);
   start = omp_get_wtime();
//   #pragma omp parallel
//   {
//	   #pragma omp single
//	   {
	   //_traverse_parallel(0,par_data);
   traverse_parallel(par_data);
//	   }
//   }

   sum=0;
   for (int i=0; i<10; i++){
	   sum += par_data[i];
	   printf("par_data[%d] %d\n", i , par_data[i]);
   }
   end = omp_get_wtime();
   printf("sum parallel is %d, ", sum);
   printf("time is %lf\n\n", end - start);

   return 0;

}
