#include <stdio.h>

#define TOTAL_THREADS 172032 
__global__ void dotProduct(double* d_c, double* d_a, double* d_b, int length, int valsPerThread) {
   // declare a buffer in shared memory to hold the partial reductions from each thread
   // in a block. You will need to use a constant value to declare this, so use the
   // number of threads/block that you have computed. 
   . . . 

   // declare a thread local/automatic variable (we'll call it c) in a register to hold
   // the results for each thread in the loop below.
   . . . 

   // compute the local dot product for each thread's values
   // each thread will do a multiply and summation across valsPerThread elements
   // of the d_a and d_b vectors. All threads in a block should access adjacent 
   // elements. I would suggest all threads on the device accessing a block of
   // data, and then moving on to the next block, and doing this a total of
   // valsPerThread times.
   for (int i=0; i<valsPerThread; i++) {
      . . .
   }

   // store c into the proper thred position of the shared memory buffer declared
   // above.
   . . . 

   // reduce the values in the buffer to have a single value in the zero element of
   // each buffer.  Use the "good" reduction described in the histogram slides
   // Remember to synchronize appropriately.
   . . . 

   // write the partial reduction for each block stored in element zero of the shared 
   // buffer, i.e., the value produced by the reduction above, into the proper
   // location for the block in d_c.
   . . . 
}

double hdotProduct(double* h_c, double* h_a, double* h_b, int lengthBytes, int lengthElements,
                  int outputSize, int numBlocks, int threadsBlock) {
   double *d_a, *d_b, *d_c; 

   // Allocate memory on the device for the d_a, d_b and d_c arrays. Note that the
   // lengths of each are in bytes, not doubles.
   //
   // Copy the h_a and h_b arrays to the d_a and d_b arrays on the gpu.  There is no 
   // need to copy d_c values as d_c only holds return values.
   . . . 

   // launch the kernel. Have four warps of 32 threads (128 threads) for each block.
   // If you use print statements make sure to have a cudaDeviceSynchronize();
   // statement after the launch.
   . . . 

   // copy the d_c array from the device into the h_c array.
   // free d_a, d_b and d_c.
   . . . 

   // sum the values now in h_c to get the final reduction value, and return that from
   // the function.
   . . . 
}

int main(int argc, char** args) {

   // compute necessary values for the problem, such as number threads per block, etc..
   . . . 

   // declare and allocate h_a, h_b and h_c on the host. 
   . . . 

   // initialize h_a and h_b. I initialized one with i, the position in the array 
   // being initialized, and the other with 1.
   . . . 

   // compute and print the sequential solution
   . . . 

   // call hdotProduct, print the value of c returned (which should equal the sequential
   // value printed above, and free h_a, h_b and h_c.
   . . . 
}
   
