#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[]) {

  int i; 
  #pragma omp parallel for 
  for (i=0; i < omp_get_num_threads( ); i++) {
     /* Obtain thread number */
     int tid = omp_get_thread_num();

     #pragma omp critical 
     printf("tid: %d, tid address: %p\n",tid, (void*) &tid); 
  }  /* All threads join master thread and disband */
}

