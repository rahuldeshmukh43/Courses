
#include <math.h>
#include "tiff.h"
#include "allocate.h"
#include "randlib.h"
#include "typeutil.h"

void error(char *name);
int32_t clip(double pixel);

int main (int argc, char **argv) 
{
  FILE *fp;
  struct TIFF_img input_img, color_img;
  double h_mn;
  int32_t i,j,k,m,n;

  if ( argc != 2 ) error( argv[0] );

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
  if ( input_img.TIFF_type != 'c' ) {
    fprintf ( stderr, "error:  image must be 24-bit color\n" );
    exit ( 1 );
  }

  /* set up structure for output color image */
  /* Note that the type is 'c' rather than 'g' */
  get_TIFF ( &color_img, input_img.height, input_img.width, 'c' );

  /* Filter image along horizontal direction */
  for ( k = 0; k < 3; k++)
  for ( i = 0; i < input_img.height; i++ )
  for ( j = 0; j < input_img.width ; j++ ) {
	  if( ( (i>=4) && (i < input_img.height-4) ) &&\
		  ( (j>=4) && (j < input_img.width-4 ) ) ){
		  h_mn=0.0;
		  for( m=-4; m<=4; m++)
		  for( n=-4; n<=4; n++){
		  h_mn += input_img.color[k][i+m][j+n];
		  }
		  h_mn /= 81.0;
		  color_img.color[k][i][j] = clip(h_mn);
	  }
	  else{
		  color_img.color[k][i][j] = 0;
	  }
  }

  /* open color image file */
  if ( ( fp = fopen ( "color.tif", "wb" ) ) == NULL ) {
      fprintf ( stderr, "cannot open file color.tif\n");
      exit ( 1 );
  }
    
  /* write color image */
  if ( write_TIFF ( fp, &color_img ) ) {
      fprintf ( stderr, "error writing TIFF file %s\n", argv[2] );
      exit ( 1 );
  }
    
  /* close color image file */
  fclose ( fp );

  /* de-allocate space which was used for the images */
  free_TIFF ( &(input_img) );
  free_TIFF ( &(color_img) );

  return(0);
}

int32_t clip(double pixel_double){
	int32_t p;
	p = ( int32_t ) pixel_double;
	if(pixel_double>255.0){return 255;}
	else{
		if(pixel_double<0.0){return 0;}
		return p;
	}
}

void error(char *name)
{
    printf("usage:  %s  image.tiff \n\n",name);
    printf("this program reads in a 24-bit color TIFF image.\n");
    printf("It then horizontally filters the green component, adds noise,\n");
    printf("and writes out the result as an 8-bit image\n");
    printf("with the name 'green.tiff'.\n");
    printf("It also generates an 8-bit color image,\n");
    printf("that swaps red and green components from the input image");
    exit(1);
}
