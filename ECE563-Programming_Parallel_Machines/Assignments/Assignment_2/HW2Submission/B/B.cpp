#include <stdio.h>
#include <unistd.h>
#include <omp.h>
using namespace std;

int SIZE=1000000;
//int SIZE=100;

void init_array(int* arr, int N, int val=0){
	/* initialize in parallel */
	#pragma omp parallel for
	for(int i=0; i<N; i++){
		arr[i] = val;
	}
	return;
}

int omp_thread_count() {
	/* for counting num threads in case of gcc */
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

int sequential_sum(int* arr, int N){
	int sum = 0;
	for(int i=0; i<N; i++){
		sum += arr[i];
	}
	return sum;
}

int manual_reduction_sum(int* arr, int N, int num_threads){
	int res[num_threads*8];
	int sum=0;

	init_array(res, num_threads*8, 0);

	//parallel
	#pragma omp parallel for
	for(int i=0; i<N; i++){
		int mythread=omp_get_thread_num();
		res[mythread*8] += arr[i];
	}

	// serial part
	for(int i=0; i<num_threads*8; i++ ){
		sum += res[i];
	}
	return sum;
}

int omp_reduction_sum(int* arr, int N){
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i=0; i<N; i++){
    	sum += arr[i];
    }
    return sum;
}


int main(){
	// initialize variables
	int num_threads;
	int arr[SIZE], sum;
	double start, end;

	num_threads=omp_thread_count();

	/* Initialize the array with zeros*/
	start = omp_get_wtime();
	init_array(arr, SIZE, 1);
	end = omp_get_wtime();
	printf("Time to initialize: %lf\n\n", end - start);

	/* Perform a sequential sum reduction
	 * and time it
	 */
	start = omp_get_wtime();
	sum = sequential_sum(arr, SIZE);
	end = omp_get_wtime();
	printf("Sequential sum: %d\n", sum);
	printf("Time for sequential sum: %lf\n\n", end- start);

	/* Perform a reduction using omp manually*/
	start = omp_get_wtime();
	sum = manual_reduction_sum(arr,SIZE, num_threads);
	end = omp_get_wtime();
	printf("Manual reduction sum: %d\n", sum);
	printf("Time for manual reduction sum: %lf\n\n", end- start);

	/* Perform a reduction using omp*/
	start = omp_get_wtime();
	sum = omp_reduction_sum(arr, SIZE);
	end = omp_get_wtime();
	printf("OpenMP reduction sum: %d\n", sum);
	printf("Time for OpenMP reduce sum: %lf\n\n", end- start);


	return 0;

}
