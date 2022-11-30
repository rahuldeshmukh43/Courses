#include "stdio.h"

__global__ void mykernel( ) {
   printf("Hello world from the device, block %d, thread %d!\n", blockIdx.x, threadIdx.x); 
}


int main(void) {
   mykernel<<<3,2>>>();
   cudaDeviceSynchronize( );
   printf("Hello World from the host!!\n");  return 0;
}
