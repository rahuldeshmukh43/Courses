#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

#define NUMNODES 100
#define NUMSTEPS 1000
#define ICETEMP 0
#define INITRODTEMP 100
/*
 * Heat Conduction Problem: Boundary Value Problem
 *
 * t=t0 0C x-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-x 0C // all init 100 x=0
 * t=t1 0C x-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-x 0C	// solve for o's
 * t=t2 0C x-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-x 0C // solve for o's
 * t=t3 0C x-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-x 0C // solve for o's
 *
 * Update equation:
 * 		P_{t,n} = (P_{t-1,n-1}+P_{t-1,n}+P_{t-1,n+1})/3
 *
 *
 */

#define TIMING
#define DEBUG

void Check_memory_allocation(void *ptr){
	if(ptr==NULL){
		printf("MEMORY ERROR: Cannot dynamically allocate on heap\n");
		exit(1);
	}
}

void InitGrid(float **grid){
	int t, x;

	//init x=0 and x=NUMNODES-1 as ICETEMP for all t
	for(t=0; t<NUMSTEPS; t++){
		grid[t][0] = ICETEMP;
		grid[t][NUMNODES-1] = ICETEMP;
	}

	//init t=0 as INITRODTEMP
	for(x=1; x<NUMNODES-1; x++){
		grid[0][x] = INITRODTEMP;
	}
	return;
}

float** Create_Grid(){
	float **grid;
	grid = new float*[NUMSTEPS];
	Check_memory_allocation(grid);

	for(int t=0; t<NUMSTEPS; t++){
		grid[t] = new float[NUMNODES];
		Check_memory_allocation(grid[t]);
	}

	InitGrid(grid);
	return grid;
}

void Print_Grid_stdout(float **grid){
//	for(int t=0; t<NUMSTEPS; t++){
	int t = NUMSTEPS-1;
	for(int x=0; x<NUMNODES; x++){
		if(x==NUMNODES-1){
			printf("%2.6f\n", grid[t][x]);
			continue;
		}
		printf("%2.6f, ", grid[t][x]);
	}
//	}
	return;
}

void Print_Grid(FILE *fp, float **grid){
	for(int t=0; t<NUMSTEPS; t++){
		for(int x=0; x<NUMNODES; x++){
			if(x==NUMNODES-1){
				fprintf(fp, "%2.6f\n", grid[t][x]);
				continue;
			}
			fprintf(fp, "%2.6f,", grid[t][x]);
		}
	}
	return;
}

void ODE_update(float **grid, int t, int x){
/*	Update equation:
 *	P_{t,x} = (P_{t-1,x-1}+P_{t-1,x}+P_{t-1,x+1})/3
 */
	grid[t][x] = (grid[t-1][x-1] + grid[t-1][x] + grid[t-1][x+1])/3.0;
	return;
}

void Solve_time_step(float **grid, int t){
	// solve for time step t
	for(int x=1; x<NUMNODES-1; x++){
		ODE_update(grid, t, x);
	}
	return;
}

bool Is_equal(float **grid1, float **grid2){
	bool flag=true;
	#pragma omp parallel for reduction(&&:flag)
	for(int i=0; i<NUMNODES*NUMSTEPS; i++){
		int t = i / NUMNODES; /// same time
		int x = i % NUMNODES;
		flag &= ( grid1[t][x] == grid2[t][x]);
	}
	return flag;
}

int main(){
	float **serial_grid, **par_grid;
	int t;
	double start, end;
	FILE *fp_serial, *fp_par;

	//allocate and init grid
	serial_grid = Create_Grid();
	par_grid = Create_Grid();

	//solve
	//----------------- serial --------------------
	start = omp_get_wtime();
	for(t=1; t<NUMSTEPS; t++){
		Solve_time_step(serial_grid, t);
	}
	end = omp_get_wtime();
	#ifdef TIMING
	printf("Time for serial: %lf\n", end - start);
	#endif

	Print_Grid_stdout(serial_grid);

	fp_serial = fopen("serial.csv","w");
	Print_Grid( fp_serial, serial_grid );
	fclose(fp_serial);
	printf("\n");
	//----------------- parallel --------------------
	start = omp_get_wtime();
	for(t=1; t<NUMSTEPS; t++){
		#pragma omp parallel
		Solve_time_step(par_grid, t);
	}
	end = omp_get_wtime();
	#ifdef TIMING
	printf("Time for parallel: %lf\n", end - start);
	#endif

	Print_Grid_stdout(serial_grid);

	fp_par = fopen("parallel.csv","w");
	Print_Grid( fp_par, par_grid );
	fclose(fp_par);

	bool flag_equal = Is_equal(serial_grid, par_grid);
	printf("\nThe two solutions are same: %s\n", flag_equal?"True":"False");

	return 0;
}
