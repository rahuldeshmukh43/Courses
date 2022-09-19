#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

#include <algorithm>
#include <limits>
#include <random>


using namespace std;

// HW3 Part 3

//int DIM=10000;
int MIN=-10;
int MAX=10;

long int dot(vector<int> a, vector<int> b ){
	long int sum=0;
	#pragma omp parallel for reduction(+:sum)
	for(int i=0; i<a.size(); i++){
		sum += a[i]*b[i];
	}
	return sum;
}

double norm(vector<int> a){
	long int sum=0;
	#pragma omp parallel for reduction(+:sum)
	for(int i=0; i<a.size(); i++)
	{
		sum += a[i]*a[i];
	}
	return sqrt(sum);
}

void print_vec(vector<int> a){
	for(int i=0; i<a.size(); i++)
	{
		printf("%d ",a[i]);
		}
	printf("\n");
	return;
}

// random vector generator
// cite: https://stackoverflow.com/a/32887614/8645905
static vector<int> generate_data(size_t size)
{
    using value_type = int;
    // We use static in order to instantiate the random engine
    // and the distribution once only.
    // It may provoke some thread-safety issues.
    static uniform_int_distribution<value_type> distribution(
        MIN,
        MAX);
    static default_random_engine generator;

    vector<value_type> data(size);
    generate(data.begin(), data.end(), []() { return distribution(generator); });
    return data;
}

double sequential_cosine_dist(vector<int>a, vector<int>b){
	int i;
	long int a_dot_b=0;
	double a_norm=0., b_norm=0.;
	double cos_dist;
	for(i=0; i<a.size();i++)
	{
		a_dot_b += a[i]*b[i];
		a_norm += a[i]*a[i];
		b_norm += b[i]*b[i];
	}
	a_norm= sqrt(a_norm);
	b_norm= sqrt(b_norm);
	/*
	printf("a_dot_b: %ld\n",a_dot_b);
	printf("a_norm: %0.6f\n",a_norm);
	printf("b_norm: %0.6f\n",b_norm);
	*/
	cos_dist = (double)a_dot_b/( a_norm * b_norm);
	return cos_dist;
}

int main(int argc, char *argv[]){
	/*
	 *  Given two vectors, we compute cosine distance
	 *  between them using parallel sections.
	 *  The first section will compute dot product
	 *  second an third section will compute norm
	 *  Finally master returns the cosine distance
	 *  cos(theta) = <a,b>/(||a|| * ||b||)
	 *
	 */

	//parse args
	if(argc<1){
		printf("Provide size of vector as arg");
		return 1;
	}
	int DIM = atoi(argv[1]);

	vector<int> a(DIM), b(DIM); // zero initialized
	long int a_dot_b;
	double a_norm, b_norm;
	double cos_dist;

	// random init vectors
	a = generate_data(DIM);
	b = generate_data(DIM);
	// print vectors
	printf("Vector A:\n");
	print_vec(a);
	printf("\nVector B:\n");
	print_vec(b);
	// pragma sections
	#pragma omp parallel sections
	{
		//dot
		#pragma omp section
		{
			a_dot_b = dot(a,b);
		}

		//norm a
		#pragma omp section
		{
			a_norm = norm(a);
		}

		//norm b
		#pragma omp section
		{
			b_norm = norm(b);
		}
	}
	/*
	printf("a_dot_b: %ld\n",a_dot_b);
	printf("a_norm: %0.6f\n",a_norm);
	printf("b_norm: %0.6f\n",b_norm);
	*/
	//cosine distance
	cos_dist = (double)a_dot_b/( a_norm * b_norm);

	printf("\nCosine Distance: %0.6f\n", cos_dist);

	/*sequential
	cos_dist = sequential_cosine_dist(a, b);
	printf("Sequential Cosine Distance: %0.6f\n", cos_dist);
	*/
	return 0;
}



