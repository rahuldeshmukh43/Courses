#include <stdio.h>
#include <unistd.h>
#include <omp.h>
using namespace std;

int N=10000000;

int main(){
	float sum=0.0;
	int i;

	// first loop
	for(i=1; i<=N; i++){
		sum += 1.0/float(i);
	}
	printf("First loop sum: %f\n", sum);

	//second loop
	sum=0.0;
	for(i=N; i>0; i--){
		sum += 1.0/float(i);
	}
	printf("Second loop sum: %f\n", sum);

	//omp reduce
	sum=0.0;
	#pragma omp parallel for reduction(+:sum) //schedule(static)
	for(i=1; i<=N; i++){
		sum += 1.0/float(i);
	}
	printf("OpenMP loop sum: %f\n", sum);

	return 0;
}
