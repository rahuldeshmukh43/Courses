#include <stdio.h>
#include <unistd.h>
#include <omp.h>

//int omp_thread_count() {
//	/* for counting num threads in case of gcc */
//    int n = 0;
//    #pragma omp parallel reduction(+:n)
//    n += 1;
//    return n;
//}

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

int main(){

	int id, nthreads, nproc;
	int len=30;
	char hostname[30];

	/* get name of host */
	gethostname(hostname, len);

	/* Master thread */
	//nthreads = omp_get_num_threads(); //does not work in case of g++/gcc
	nthreads = omp_thread_count();
	nproc = omp_get_num_procs();
	id = omp_get_thread_num();
	printf("Num Threads: %d Num Proc: %d on host: %s\n", nproc, nthreads, hostname);
	printf("SERIAL REGION: Master thread id: %d\n",id);

	/* OMP parallel region */
	#pragma omp parallel private(id)
	{
		id = omp_get_thread_num();
		printf("PARALLEL REGION: Worker thread id: %d\n", id);
	}
	/* barrier */
	printf("SERIAL REGION: Finished!\n");
	return 0;
}
