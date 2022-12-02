#include <stdio.h>
#include "cuda.h"
#include <omp.h>
#include "iostream"

using namespace std;

#define VEC_LEN_MUL 5

//V100 params Volta
//#define NUM_SM 80 //84
//#define MAX_THREADS_PER_SM 2048
//#define MAX_NUM_BLKS_PER_SM 32
//#define MAX_NUM_WARPS_PER_SM 64
//#define MAX_NUM_THREADS 172032
//#define WARP_SIZE 32

//#define BLOCK_SIZE 128 // i choose this

//a5000 params Ampere
#define NUM_SM 64
#define MAX_THREADS_PER_SM 1536
#define MAX_NUM_BLKS_PER_SM  //?
#define MAX_NUM_WARPS_PER_SM 48 // = MAX_THREADS_PER_SM/ WARP_SIZE
#define MAX_NUM_THREADS 98304
#define WARP_SIZE 32

#define BLOCK_SIZE 128// i choose this

void deviceQuery ()
{
  cudaDeviceProp prop;
  int nDevices=0, i;
  cudaError_t ierr;

  ierr = cudaGetDeviceCount(&nDevices);
  if (ierr != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierr)); }

  printf("#----------------------------------------------------#\n");
  printf("\t\t GPU Specs\n");
  printf("#----------------------------------------------------#\n");
  for( i = 0; i < nDevices; ++i )
  {
     ierr = cudaGetDeviceProperties(&prop, i);
     printf("Device number: %d\n", i);
     printf("  Device name: %s\n", prop.name);
     printf("  Compute capability: %d.%d\n\n", prop.major, prop.minor);

     printf("  Clock Rate: %d kHz\n", prop.clockRate);
     printf("  Total SMs: %d \n", prop.multiProcessorCount);
     printf("  Shared Memory Per SM: %lu bytes\n", prop.sharedMemPerMultiprocessor);
     printf("  Registers Per SM: %d 32-bit\n", prop.regsPerMultiprocessor);
     printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
//     printf("  Max num thread blocks that can reside on a SM: %d", prop.maxBlocksPerMultiProcessor);
     printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
     printf("  Total Global Memory: %lu bytes\n", prop.totalGlobalMem);
     printf("  Memory Clock Rate: %d kHz\n\n", prop.memoryClockRate);

     printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
     printf("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
     printf("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
     printf("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);

     printf("  Max blocks in X-dimension of grid: %d\n", prop.maxGridSize[0]);
     printf("  Max blocks in Y-dimension of grid: %d\n", prop.maxGridSize[1]);
     printf("  Max blocks in Z-dimension of grid: %d\n\n", prop.maxGridSize[2]);

     printf("  Shared Memory Per Block: %lu bytes\n", prop.sharedMemPerBlock);
     printf("  Registers Per Block: %d 32-bit\n", prop.regsPerBlock);
     printf("  Warp size: %d\n", prop.warpSize);
     printf("#----------------------------------------------------#\n\n");
  }
}


__global__ void dotProduct(double* d_c, double* d_a, double* d_b, int length, int valsPerThread)
{
   // declare a buffer in shared memory to hold the partial reductions from each thread
   // in a block. You will need to use a constant value to declare this, so use the
   // number of threads/block that you have computed. 
   __shared__ double private_d_c[BLOCK_SIZE]; //private per block

   // declare a thread local/automatic variable (we'll call it c) in a register to hold
   // the results for each thread in the loop below.
   double c= 0;
   int tid = threadIdx.x;
   int gtid = blockDim.x * blockIdx.x + threadIdx.x;

   // compute the local dot product for each thread's values
   // each thread will do a multiply and summation across valsPerThread elements
   // of the d_a and d_b vectors. All threads in a block should access adjacent 
   // elements. I would suggest all threads on the device accessing a block of
   // data, and then moving on to the next block, and doing this a total of
   // valsPerThread times.
   for (unsigned int i=0; i<valsPerThread; i++) {
	   __syncthreads();
	  unsigned int j =  gridDim.x*blockDim.x * i + gtid;
	  if(j < length)
	  {
		  c += d_a[j] * d_b[j];
	  }
   }
   __syncthreads(); // block level sync

   // store c into the proper thred position of the shared memory buffer declared
   // above.
   private_d_c[tid] = c;

   // reduce the values in the buffer to have a single value in the zero element of
   // each buffer.  Use the "good" reduction described in the histogram slides
   // Remember to synchronize appropriately.
   for(unsigned int stride= blockDim.x/2; stride>0; stride /= 2)
   {
	   __syncthreads();
	   if( tid < stride )
	   {
		   private_d_c[tid] += private_d_c[ tid + stride];
	   }
   }
   __syncthreads();

   // write the partial reduction for each block stored in element zero of the shared 
   // buffer, i.e., the value produced by the reduction above, into the proper
   // location for the block in d_c.
   if( tid == 0 )
   {
	   d_c[blockIdx.x] = private_d_c[0];
   }
}

double hdotProduct(double* h_c, double* h_a, double* h_b, int lengthBytes, int lengthElements,
                  int outputSize, int numBlocks, int threadsBlock)
{
   double *d_a, *d_b, *d_c; 

   // Allocate memory on the device for the d_a, d_b and d_c arrays. Note that the
   // lengths of each are in bytes, not doubles.
	cudaError_t err = cudaMalloc((void **) &d_a, lengthBytes);
	if(err!=cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &d_b, lengthBytes);
	if(err!=cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &d_c, outputSize);
	if(err!=cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


   // Copy the h_a and h_b arrays to the d_a and d_b arrays on the gpu.  There is no 
   // need to copy d_c values as d_c only holds return values.
   cudaMemcpy(d_a, h_a, lengthBytes, cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, h_b, lengthBytes, cudaMemcpyHostToDevice);

   // launch the kernel. Have four warps of 32 threads (128 threads) for each block.
   // If you use print statements make sure to have a cudaDeviceSynchronize();
   // statement after the launch.
   int valsPerThread = lengthElements/(numBlocks*BLOCK_SIZE);
   dim3 DimGrid(numBlocks, 1, 1);
   dim3 DimBlock(threadsBlock, 1, 1);
   dotProduct<<<DimGrid, DimBlock>>>(d_c, d_a, d_b, lengthElements, valsPerThread);//TODO: what is valsperthread
   cudaDeviceSynchronize( );


   // copy the d_c array from the device into the h_c array.
   // free d_a, d_b and d_c.
   cudaMemcpy(h_c, d_c, outputSize, cudaMemcpyDeviceToHost);
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

   // sum the values now in h_c to get the final reduction value, and return that from
   // the function.
   double c = 0.;
   #pragma omp parallel for reduction( + : c)
   for(int i=0; i < numBlocks ; i++)
   {
	   c += h_c[i];
   }

   return c;
}

int main(int argc, char** args)
{

   // compute necessary values for the problem, such as number threads per block, etc..
   int lengthBytes, lengthElements, outputSize, numBlocks, threadsBlock;
   lengthElements = MAX_NUM_THREADS*VEC_LEN_MUL;
   lengthBytes = lengthElements*sizeof(double);
   threadsBlock = BLOCK_SIZE;
   numBlocks = MAX_NUM_THREADS/threadsBlock;
   outputSize = numBlocks*sizeof(double);

   // declare and allocate h_a, h_b and h_c on the host.
   double *h_a = (double*) malloc(lengthBytes);
   double *h_b = (double*) malloc(lengthBytes);
   double *h_c = (double*) malloc(outputSize);

   // initialize h_a and h_b. I initialized one with i, the position in the array 
   // being initialized, and the other with 1.
   #pragma omp parallel for
   for(int i=0; i < lengthElements; i++)
   {
	   h_a[i] = (double) i;
	   h_b[i] = 1.;
   }

   // compute and print the sequential solution
   double seq_c = 0.;
   #pragma omp parallel for reduction( + : seq_c)
   for(int i=0; i < lengthElements; i++)
   {
	   seq_c += h_a[i] * h_b[i];
   }

   // call hdotProduct, print the value of c returned (which should equal the sequential
   // value printed above, and free h_a, h_b and h_c.
   double gpu_c;
   gpu_c = hdotProduct(h_c, h_a, h_b, lengthBytes, lengthElements,
		   outputSize, numBlocks, threadsBlock);

   deviceQuery();

   printf("\nFor GPU execution: \n");
   printf("\t block size: %d\n", threadsBlock);
   printf("\t num blocks: %d\n\n", numBlocks);

   printf("Sequential dotproduct: %f\n",seq_c);
   printf("GPU dotproduct: %f\n\n",gpu_c);
   if (gpu_c == seq_c) printf("Yes! Sequential and GPU sum is same\n\n");
   else printf("!! Error: Sequential and GPU sum is different\n\n");


   free(h_a);
   free(h_b);
   free(h_c);
}
   
