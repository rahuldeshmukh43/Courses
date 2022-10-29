#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
// KILL change int to double , MPI_INTEGER to MPI_DOUBLE

#define ARRAY_ROWS 1600
#define ARRAY_COLS 1600
// #define ARRAY_ROWS 16
// #define ARRAY_COLS 16
#define SQRT_P 4
#define PROC_ROWS SQRT_P
#define PROC_COLS SQRT_P
#define LOCAL_ARRAY_ROWS ARRAY_ROWS/PROC_ROWS
#define LOCAL_ARRAY_COLS ARRAY_COLS/PROC_COLS
#define SHIFT_COLUMN 0
#define SHIFT_ROW 1

typedef struct Coord {
   int row;
   int col;
} CoordT;

CoordT pid2RowCol[SQRT_P*SQRT_P];
int rowCol2pid[SQRT_P][SQRT_P];

int max(int a, int b) {
   return (a < b) ? b : a;
}

void printArray(double* a, int rows, int cols, int sparseness) {
   printf("rows: %d, cols: %d, sparseness: %d\n", rows, cols, sparseness);
   fflush(stdout);
   for (int i=0; i<rows; i+=sparseness) {
      for (int j=0; j<cols; j+=sparseness) {
         printf("%.2f ", *(a + i*cols + j));
      }
      printf("\n");
   }
   printf("\n\n\n");
}

double* makeArray(int rows, int cols, int pid, int zero) {
   double* arr = (double*) malloc(rows*cols*sizeof(double));
   if (arr == NULL) printf("bad allocation"); fflush(stdout);

   int pidRow = pid2RowCol[pid].row;
   int pidCol = pid2RowCol[pid].col;
   int elt=0;
   for (int r=0; r<rows; r++) {
      for (int c=0; c<cols; c++) {
         // *(arr + r*cols + c) = pidRow*100000+pidCol*1000+1;
         if (zero==0) {
            *(arr + r*cols + c) = 0;
         } else if (zero==-1) {
            *(arr + r*cols + c) = (pidRow*LOCAL_ARRAY_ROWS+r) + (pidCol*LOCAL_ARRAY_COLS+c);
         } else {
            *(arr + r*cols + c) = zero;
         } 
      }
   }

   return arr;
}

void initRowColPidMaps( ) {
   int pcount=0;
   for (int r=0; r<SQRT_P; r++) {
      for (int c=0; c<SQRT_P; c++) {
         pid2RowCol[pcount].row = r;
         pid2RowCol[pcount].col = c;
         rowCol2pid[r][c] = pcount;
         pcount++;
      }
   }
}

MPI_Datatype makeType( ) {
   // define the data type here
   MPI_Datatype block;
   MPI_Type_vector(1, LOCAL_ARRAY_ROWS*LOCAL_ARRAY_COLS, 0, MPI_DOUBLE, &block);
   MPI_Type_commit(&block);
   return block;
}

int calculateShift(int current, int shiftAmount, int max) { // shift always up or left
   int base = current-shiftAmount;
   return (base < 0) ? base+max : base;
}

void shift(double* ary, double* buffer, int shiftAmount, int shiftRow, int pid, 
           MPI_Datatype datatype) {

   if (shiftAmount==0) return;

   int sendTo[SQRT_P];
   int recvFrom[SQRT_P];
   
   for (int src=0; src<SQRT_P; src++) {
      int dest = calculateShift(src, shiftAmount, SQRT_P);
      sendTo[src] = dest;
      recvFrom[dest] = src;
   }

   int pidRow = pid2RowCol[pid].row;
   int pidCol = pid2RowCol[pid].col;

   MPI_Request request;
   MPI_Status status;
             
   int srcPid = shiftRow ? rowCol2pid[recvFrom[pidRow]][pidCol] :
                           rowCol2pid[pidRow][recvFrom[pidCol]];
                          
   int check = 0;
   if (srcPid != pid) {
      MPI_Irecv(buffer, 1, datatype, srcPid, 1, MPI_COMM_WORLD, &request);
   } 

   int destPid = shiftRow ? rowCol2pid[sendTo[pidRow]][pidCol] :
                           rowCol2pid[pidRow][sendTo[pidCol]];
   if (destPid != pid) {
      MPI_Send(ary, 1, datatype, destPid, 1, MPI_COMM_WORLD);
   }

   if (pid != srcPid) {
      MPI_Wait(&request, &status); 
      memcpy(ary, buffer, LOCAL_ARRAY_ROWS*LOCAL_ARRAY_COLS*sizeof(double));
   }
}

int main (int argc, char *argv[]) {

   int pid;
   int numP = SQRT_P*SQRT_P;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   MPI_Comm_size(MPI_COMM_WORLD, &numP);

   MPI_Datatype blockType = makeType( );

   if (numP != SQRT_P*SQRT_P) {
      printf("numP should be %d, is: %d. Terminating\n", SQRT_P*SQRT_P, numP);
      fflush(stdout);
      MPI_Finalize( );
      return -1;
   }

   initRowColPidMaps( );

   double* a = makeArray(LOCAL_ARRAY_ROWS, LOCAL_ARRAY_COLS, pid, -1);
   double* b = makeArray(LOCAL_ARRAY_ROWS, LOCAL_ARRAY_COLS, pid, -1);
   double* sendBuffer = makeArray(LOCAL_ARRAY_ROWS, LOCAL_ARRAY_COLS, pid, -44);
   double* c = makeArray(LOCAL_ARRAY_ROWS, LOCAL_ARRAY_COLS, pid, 0);

   MPI_Barrier(MPI_COMM_WORLD);

   double elapsedTime = -MPI_Wtime( );
   shift(a, sendBuffer, pid2RowCol[pid].row, SHIFT_COLUMN, pid, blockType);
   shift(b, sendBuffer, pid2RowCol[pid].col, SHIFT_ROW, pid, blockType);

   for (int iter=0; iter<SQRT_P; iter++) {
      for (int i=0; i<LOCAL_ARRAY_ROWS; i++) {
         for (int j=0; j<LOCAL_ARRAY_COLS; j++) {
            int comp = 0.;
            for (int k=0; k<LOCAL_ARRAY_ROWS; k++) {
               comp += *(a + i*LOCAL_ARRAY_COLS + k) * *(b + k*LOCAL_ARRAY_COLS + j);
            }
            *(c + i*LOCAL_ARRAY_COLS + j) += comp;
         }
      }
      if (iter != SQRT_P-1) {
         shift(a, sendBuffer, 1, SHIFT_COLUMN, pid, blockType);
         shift(b, sendBuffer, 1, SHIFT_ROW, pid, blockType);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   double timeTaken = (elapsedTime + MPI_Wtime( ));
   if (pid == 0) { 
      printf("time take for %dx%d multiply is: %f\n", ARRAY_ROWS, ARRAY_COLS, timeTaken);
   }

   if (pid==0) {
      printf("array values: \n");
      printArray(c, LOCAL_ARRAY_ROWS, LOCAL_ARRAY_COLS, max(1, LOCAL_ARRAY_ROWS/16));
   }

   MPI_Finalize( );
   return 0;
}
