#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) {
	/* print count using single, master,
	 * critical
	 */
	int crit_count=0, single_count=0, master_count=0;
	#pragma omp parallel
	{
		#pragma omp single
		single_count++;

		#pragma omp single
		master_count++;

    	#pragma omp critical
		crit_count++;
	}
	printf("Count of passes through sections:\n");
	printf("master: %d\n", master_count);
	printf("single: %d\n", single_count);
	printf("critical: %d\n", crit_count);
	return 0;
}
