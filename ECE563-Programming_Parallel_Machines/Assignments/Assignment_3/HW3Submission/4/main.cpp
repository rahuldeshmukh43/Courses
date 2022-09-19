#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// HW3 Part 4

int main (int argc, char *argv[]) {
	int i, tid;
	printf("pragma omp parallel\n");
	#pragma omp parallel
	{
		for(int i=0;i<20;i++){
			tid = omp_get_thread_num();
			printf("i: %d tid: %d\n",i, tid);
		}
	}

	printf("\npragma omp parallel for\n");
	#pragma omp parallel for
	for(int i=0;i<20;i++){
		tid = omp_get_thread_num();
		printf("i: %d tid: %d\n",i,tid);
	}

	return 0;
}
