
#include <math.h>
#include "tiff.h"
#include "allocate.h"
#include "randlib.h"
#include "typeutil.h"
#include "iostream"
using namespace std;

void error(char *name);
int median_filter(uint8_t **X, char **wts,int filter_wd, int row, int col);
int* Sort(int *A, int *sorted_ids, int size);
void merge_sort(int *A, int *sorted_ids, int start, int end);
void merge(int *A,int *sorted_ids,int start,int mid,int end);
void print_array(int *A, int size);

int32_t filter_size=5;

int main (int argc, char **argv) 
{
  FILE *fp;
  struct TIFF_img input_img, out_img;
  char **wts;
  int32_t i,j;

  if ( argc != 3 ) error( argv[0] );

  /* open image file */
  if ( ( fp = fopen ( argv[1], "rb" ) ) == NULL ) {
    fprintf ( stderr, "cannot open file %s\n", argv[1] );
    exit ( 1 );
  }

  /* read image */
  if ( read_TIFF ( fp, &input_img ) ) {
    fprintf ( stderr, "error reading file %s\n", argv[1] );
    exit ( 1 );
  }

  /* close image file */
  fclose ( fp );

  /* check the type of image data */
  if ( input_img.TIFF_type != 'g' ) {
    fprintf ( stderr, "error:  image must be 8-bit color\n" );
    exit ( 1 );
  }

  /* Allocate image of double precision floats */
  wts = (char **)get_img(filter_size, filter_size, sizeof(char));

  /* create wts array*/
  for ( i = 0; i < filter_size; i++ ){
  for ( j = 0; j < filter_size; j++ ) {
    if((i==0)||(i==filter_size-1)||(j==0)||(j==filter_size-1)){
    	wts[i][j] =1;}
    else{wts[i][j] =2;}
  }
  }

  /* set up structure for output achromatic image */
  /* to allocate a full color image use type 'c' */
  get_TIFF ( &out_img, input_img.height, input_img.width, 'g' );

  /* median filtering of input image */
  for ( i = 0; i < input_img.height; i++ )
  for ( j = 0; j < input_img.width; j++ ) {
	  // zero for boundary pixels
	  if( (i<(filter_size/2))||(j<(filter_size/2))||\
		  (i > input_img.height-(filter_size/2)-1)||\
		  (j>input_img.width-(filter_size/2)-1) ){
		  out_img.mono[i][j] = 0;
		  continue;
	  }
	  //median filtering
	  out_img.mono[i][j] = (uint8_t)median_filter(input_img.mono, wts, filter_size, i, j);
  }
  fprintf(stdout,"filtering done\n");
  /* open output image file */
  if ( ( fp = fopen ( argv[2], "wb" ) ) == NULL ) {
    fprintf ( stderr, "cannot open file %s\n",argv[2]);
    exit ( 1 );
  }

  /* write output image */
  if ( write_TIFF ( fp, &out_img ) ) {
    fprintf ( stderr, "error writing TIFF file %s\n", argv[2] );
    exit ( 1 );
  }

  /* close output image file */
  fclose ( fp );

  /* de-allocate space which was used for the images */
  free_TIFF ( &(input_img) );
  free_TIFF ( &(out_img) );
  
  free_img( (void**)wts );

  return(0);
}

void error(char *name)
{
    printf("usage:  %s  image.tiff \n\n",name);
    printf("this program reads in a 8-bit grayscale TIFF image.\n");
    printf("it then restores the image by reducing noise using median filtering.\n");
    printf("finally it saves an 8-bit grayscale image for the restored image.\n");
    exit(1);
}

int median_filter(uint8_t **X, char **wts, int filter_wd, int row, int col){
	int i=0, j, r, c, size=filter_wd*filter_wd;
	int sum1, sum2;
	int *X_array, *a_array;
	int *sorted_ids=NULL;
	//populate array for sorting
	X_array = new int[size];
	a_array = new int[size];
	for(i=0; i< size; i++){
		r = i/filter_wd;
		c = i%filter_wd;
		X_array[i] = X[row + r -filter_wd/2][col+ c - filter_wd/2];
		a_array[i] = wts[r][c];
	}
	//sort the arrays using X_array values
	sorted_ids=Sort(X_array, sorted_ids, size);
	sorted_ids=Sort(a_array, sorted_ids, size);//sort using ids

	//find i*
	for(i=0; i<size; i++){
		sum1=0; sum2=0;
		//sum1 1-i
		for(j=0;j<i;j++){sum1+=a_array[j];}
		//sum2 i-size
		for(j=i;j<size;j++){sum2+=a_array[j];}
		//check
		if(sum1>=sum2){break;}
	}
	return X_array[i-1];
}

int* Sort(int *A, int *sorted_ids, int size){
	//sort an array
	int i=0, *temp;
	temp = new int[size];
	if(sorted_ids!=NULL){//sort A using ids
		for(i=0; i<size; i++){temp[i]=A[i];}
		for(i=0; i<size; i++){A[i] = temp[sorted_ids[i]];}
		return sorted_ids;
	}
	for(i=0; i<size; i++){temp[i]=i;}
	//sort the array
	merge_sort(A, temp, 0, size);
	return temp;
}

void merge_sort(int *A, int *sorted_ids, int start, int end){
	//merge sort
	if((end-start)==1){//base case
		return;}
	int mid=(start+end)/2;
	merge_sort(A, sorted_ids, start, mid);
	merge_sort(A, sorted_ids, mid, end);
	merge(A, sorted_ids, start, mid, end);
	return;
}

void merge(int *A,int *sorted_ids,int start,int mid,int end){
	//merge two sub arrays: A[start:mid], A[mid+1:end]
	//sort in descending order
	int i,j,k=0;
	int *temp, *temp_ids;
	temp = new int[end-start];
	temp_ids= new int[end-start];

	for(i=start, j=mid; (i<mid)||(j<end);){
		if((i<mid)&&(j<end)){//both subarrays not exhauted
			if(A[i]>A[j]){
				temp[k]=A[i];
				temp_ids[k] = sorted_ids[i];
				i++;
			}
			else{
				temp[k]=A[j];
				temp_ids[k] = sorted_ids[j];
				j++;}
			}
		else{//one of the sub-array has exhausted
			if(i==mid){
				temp[k] = A[j];
				temp_ids[k] = sorted_ids[j];
				j++;
				}
			else{//j=end
				temp[k]=A[i];
				temp_ids[k] = sorted_ids[i];
				i++;
				}
			}
		k++;
	}
	//copy back to A
	k=0;
	for(i=start;i<end;i++){
		A[i] = temp[k];
		sorted_ids[i] = temp_ids[k];
		k++;
	}
	return;
}

void print_array(int *A, int size){
	int i=0;
	for(;i<size;i++){
	fprintf(stdout,"%d,\t",A[i]);}
	fprintf(stdout,"\n");
	return;
}
