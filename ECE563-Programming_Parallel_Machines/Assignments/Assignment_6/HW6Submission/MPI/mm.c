#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//#define DEBUG
//#define LOG
//#define PRINT_OUTPUT


void printArray(double* a, int rows, int cols) {
   for(int i=0; i<rows; i++) {
      for(int j=0; j<cols; j++) {
         printf("%.2f ", *(a + i*cols + j));
      }
      printf("\n");
   }
   printf("\n\n\n");
}

double* mallocateArray(int size){
	double* arr = (double*) malloc(size*sizeof(double));
	return arr;
}

int* mallocBidx(int stripeSize){
	int* arr= (int*) malloc(stripeSize*sizeof(int));
	return arr;

}

double* makeArray(int rows, int cols) {
   double* arr = (double*) malloc(rows*cols*sizeof(double));

   for(int r=0; r<rows; r++) {
      for(int c=0; c<cols; c++) {
         *(arr + r*cols + c) = (double) (rows*c + c);
      }
   }

   return arr;
}

double* makeArray_A(int ROWS, int COLS, int stripeSize, int pid, int* Aidx){
//	double* arr = (double*) malloc(stripeSize*COLS*sizeof(double));
	double* arr = mallocateArray(stripeSize*COLS);
	int r, count, c;
	for(count=0, r=pid*stripeSize; r < (pid+1)*stripeSize; r++, count++){
		for(c=0; c<COLS; c++){
			*(arr + count*COLS + c) = (double) (ROWS*c + c);
		}
	}

	for(count=0, r=pid*stripeSize; r < (pid+1)*stripeSize; r++, count++){
		Aidx[count] = r;
	}
	return arr;
}

double* makeArray_B(int ROWS, int COLS, int stripeSize, int pid, int* Bidx){
//	double* arr = (double*) malloc(ROWS*stripeSize*sizeof(double));
	double* arr = mallocateArray(ROWS*stripeSize);
	int r, c, count;
	for(r=0; r < ROWS; r++){
		for(count=0, c=pid*stripeSize; c < (pid+1)*stripeSize; c++, count++){
			*(arr + r*stripeSize + count) = (double) (ROWS*c + c);
		}
	}

	for(count=0, c=pid*stripeSize; c < (pid+1)*stripeSize; c++, count++){
		Bidx[count]=c;
	}
	return arr;
}

void stripedMatMul( double *Aptr, double *Bptr, int *Bidx, double *Cptr, int COLS, int stripeSize){
	double sum;
	int k;
	for(int r=0; r<stripeSize; r++){
		for(int c=0; c<stripeSize; c++){
			sum = 0.;
			for(k=0; k<COLS; k++){
				sum += (*(Aptr + r*COLS + k)) * (*(Bptr+ k*stripeSize + c));
			}
			*(Cptr + r*COLS + Bidx[c]) = sum;
		}
	}
	return;
}

bool isEven(int x){
	return x%2 ? false: true;
}

void copy_double(double* from, double* to, int size){
	for(int i=0; i < size; i++){ *(to + i) = *(from + i); }
	return;
}
void copy_int(int* from, int* to, int size){
	for(int i=0; i < size; i++){ *(to + i) = *(from + i); }
	return;
}



int min(int i, int j) {
   return i<j ? i : j;
} 

int main (int argc, char *argv[]) {

   //------------------------ MPI ---------------------------------//
   int pid, dest_pid, source_pid;
   int NumProcs;
   int count=0;
   double time=0;

   const int ROWS = 1600;
   const int COLS = 1600;

   MPI_Status status_Bptr, status_Bidx;

   int tasks;
   if(argc!=2){
	   if(pid == 0) printf("Command line: %s <m>\n", argv[0]);
	   //MPI_Finalize();
	   exit(1);
   }
   tasks = atoi(argv[1]);
   
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &NumProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);

   if(tasks==1 && pid==0){
	//---- sequential code ---//
	double* a = makeArray(ROWS, COLS);
	double* b = makeArray(ROWS, COLS);
	double* c = makeArray(ROWS, COLS);

//	clock_t timer = -clock( );
//	for (int t=0; t<tasks; t++) {
//	   for (int i=t*stripeSize; i<min(t*stripeSize+stripeSize, ROWS); i++) {
//	      for (int j=0; j<COLS; j++) {
//	         double comp = 0.;
//	         for (int k=0; k<COLS; k++) {
//	            comp += *(a + i*COLS + k) * *(b + k*COLS + j);
//	         }
//	         *(c + i*COLS + j) = comp;
//	      }
//	   }
//	}
	
	time -= MPI_Wtime();
	for(int i=0; i<ROWS; i++){
		for(int j=0; j<COLS; j++){			
		         double comp = 0.;
		         for(int k=0; k<COLS; k++){
		            comp += *(a + i*COLS + k) * *(b + k*COLS + j);
	        	 }
		         *(c + i*COLS + j) = comp;
		}
	}
	
	//double timeTaken = (timer + clock( ))/CLOCKS_PER_SEC;
	time += MPI_Wtime();
	time = time/MPI_Wtick();
	printf("time taken for matrix multiply: %.2lf\n", time);

	#ifdef PRINT_OUTPUT	
	printArray(c, ROWS, COLS);
	#endif

	return 0;
	//------------------------//
   }

   const int stripeSize = COLS/tasks;

   #ifdef DEBUG
   printf("DEBUG: Working on pid: %d\n", pid);
   #endif

   //----- Init data A and B
   int* Aidx = mallocBidx(stripeSize);
   double* Aptr = makeArray_A(ROWS, COLS, stripeSize, pid, Aidx);
   int* Bidx = mallocBidx(stripeSize);
   double* Bptr = makeArray_B(ROWS, COLS, stripeSize, pid, Bidx);
   double* Cptr = (double*) malloc(stripeSize*COLS*sizeof(double));
   for(int i=0; i<stripeSize*COLS; i++){ *Cptr = 0.0; }

   #ifdef DEBUG
   printf("DEBUG: Matrices inited on pid: %d\n", pid);
   #endif

   //if(pid == 0){
   double* C_gathered = (double*) malloc(ROWS*COLS*sizeof(double));
   for(int i=0; i<ROWS*COLS; i++){ *(C_gathered + i) = 0; }
   //}
   
   #ifdef DEBUG
   printf("DEBUG: C_gathered inited on pid: %d\n", pid);
   #endif

   // allocate space for send and receive buffers
   double* Bptr_send = mallocateArray(ROWS*stripeSize);
   int* Bidx_send = mallocBidx(stripeSize);
   double* Bptr_recv = mallocateArray(ROWS*stripeSize);
   int* Bidx_recv = mallocBidx(stripeSize);

   #ifdef DEBUG
   printf("DEBUG: buffers allocated on pid: %d\n", pid);
   #endif


   MPI_Barrier(MPI_COMM_WORLD);
   time -= MPI_Wtime();
   //---- Compute and Communicate
   /*
    * tag --
    * 	10xx is send tag for Bptr by pid xx where xx is even
    * 	11xx is send tag for Bidx by pid xx where xx is even
    *   20xx is send tag for Bptr by pid xx where xx is odd
    *   21xx is send tag for Bidx by pid xx where xx is odd
    */
   do{
	   // Compute
	   stripedMatMul(Aptr, Bptr, Bidx, Cptr, COLS, stripeSize);
	   #ifdef DEBUG
	   printf("DEBUG: striped mat mul computed on pid: %d\n", pid);
	   #endif

	   //------------- Communicate B and Bidx
	   // ==== even procs sending to odd procs ====
	   if(isEven(pid)){
		   // copy data to send buffer
		   copy_double(Bptr, Bptr_send, ROWS*stripeSize);
		   copy_int(Bidx, Bidx_send, stripeSize);
		   #ifdef DEBUG
		   printf("DEBUG: copying data to buffers for sending on pid: %d\n", pid);
		   #endif
			
		   // send data to P+1
		   dest_pid = (pid + 1) % NumProcs;
		   MPI_Send(Bptr_send, ROWS*stripeSize, MPI_DOUBLE, dest_pid, 1000 + pid, MPI_COMM_WORLD);
		   MPI_Send(Bidx_send, stripeSize, MPI_INT, dest_pid, 1100 + pid, MPI_COMM_WORLD);
		   #ifdef DEBUG
		   printf("DEBUG: data sent from pid:%d to pid: %d\n", pid, dest_pid);
		   #endif

	   }else{
		   // recv data
		   source_pid = (pid > 0) ? (pid - 1) : NumProcs - 1;
		   #ifdef DEBUG
		   printf("DEBUG: Starting to recieve at pid %d from source pid %d\n", pid, source_pid);
		   #endif

		   MPI_Recv(Bptr_recv, ROWS*stripeSize, MPI_DOUBLE, source_pid, 1000 + source_pid, MPI_COMM_WORLD, &status_Bptr);
		   MPI_Recv(Bidx_recv, stripeSize, MPI_INT, source_pid, 1100 + source_pid, MPI_COMM_WORLD, &status_Bidx);
		   #ifdef LOG
		   printf("B: Task id:%d received %d double(s) from task %d with tag %d\n",
				   pid, ROWS*stripeSize, status_Bptr.MPI_SOURCE, status_Bptr.MPI_TAG );
		   printf("Bidx: Task id:%d received %d int(s) from task %d with tag %d\n",
				   pid, stripeSize, status_Bidx.MPI_SOURCE, status_Bidx.MPI_TAG );
		   #endif
		   // copy data to main buffer
		   copy_double(Bptr_recv, Bptr, ROWS*stripeSize);
		   copy_int(Bidx_recv, Bidx, stripeSize);
		   #ifdef DEBUG
		   printf("DEBUG: copying data from receiving buffers to application buffers on pid: %d\n", pid);
		   #endif

	   }
	   #ifdef DEBUG
	   printf("DEBUG: (pid: %d) Even(sends) Odd(recv): Done\n",pid);
	   #endif

	   // ==== even procs recv from odd procs ====
	   if(!isEven(pid)){
		   // copy data to send buffer
		   copy_double(Bptr, Bptr_send, ROWS*stripeSize);
		   copy_int(Bidx, Bidx_send, stripeSize);
		   #ifdef DEBUG
		   printf("DEBUG: copying data to buffers for sending on pid: %d\n", pid);
		   #endif

		   // send data to P+1
		   dest_pid = (pid + 1) % NumProcs;
		   MPI_Send(Bptr_send, ROWS*stripeSize, MPI_DOUBLE, dest_pid, 2000 + pid, MPI_COMM_WORLD);
		   MPI_Send(Bidx_send, stripeSize, MPI_INT, dest_pid, 2100 + pid, MPI_COMM_WORLD);
		   #ifdef DEBUG
		   printf("DEBUG: data sent from pid:%d to pid: %d\n", pid, dest_pid);
		   #endif

	   }else{
		   // recv data
		   source_pid = (pid > 0) ? (pid - 1) : NumProcs - 1;
		   MPI_Recv(Bptr_recv, ROWS*stripeSize, MPI_DOUBLE, source_pid, 2000 + source_pid, MPI_COMM_WORLD, &status_Bptr);
		   MPI_Recv(Bidx_recv, stripeSize, MPI_INT, source_pid, 2100 + source_pid, MPI_COMM_WORLD, &status_Bidx);
		   #ifdef LOG
		   printf("B: Task id:%d received %d double(s) from task %d with tag %d\n",
				   pid, ROWS*stripeSize, status_Bptr.MPI_SOURCE, status_Bptr.MPI_TAG );
		   printf("Bidx: Task id:%d received %d int(s) from task %d with tag %d\n",
				   pid, stripeSize, status_Bidx.MPI_SOURCE, status_Bidx.MPI_TAG );
		   #endif
		   // copy data to main buffer
		   copy_double(Bptr_recv, Bptr, ROWS*stripeSize);
		   copy_int(Bidx_recv, Bidx, stripeSize);
		   #ifdef DEBUG
		   printf("DEBUG: copying data from receiving buffers to application buffers on pid: %d\n", pid);
		   #endif

	   }
           #ifdef DEBUG
	   printf("DEBUG: (pid: %d)  Even(recv) Odd(sends): Done\n", pid);
	   #endif


	   //------------------------------
	   //book keeping
	   count+=1;
	   #ifdef LOG
	   printf("#----- completed step : %d / %d on pid: %d ----#\n ", count, COLS/stripeSize, pid);
	   #endif

   }while(count<COLS/stripeSize);
   MPI_Barrier(MPI_COMM_WORLD);

   //--- gather data for C --
   // gather stores data in rank order so C_gathered will be correct order
   //if(pid==0){
   MPI_Gather(Cptr, stripeSize*COLS, MPI_DOUBLE, C_gathered, stripeSize*COLS, MPI_DOUBLE, 0 , MPI_COMM_WORLD);
   //}
   MPI_Barrier(MPI_COMM_WORLD);
   time += MPI_Wtime();
   time = time/MPI_Wtick();
   MPI_Finalize();

   if(pid==0){
	printf("Total time for MPI (pid: %d world size: %d): %lf\n\n", pid, NumProcs, time);
	#ifdef PRINT_OUTPUT   
	printArray(C_gathered, ROWS, COLS);
	#endif
   }

   return 0;
}
