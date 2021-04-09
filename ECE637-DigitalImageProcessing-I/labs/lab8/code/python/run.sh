#!/bin/bash

img_dir=../..

#section 3
python simple_thresholding.py "$img_dir"/house.tif | tee sec3.log
mv ./*.tif ./output/sec3/
mv ./*.pdf ./output/sec3/
mv ./*.log ./output/sec3/
echo 'section 3 done'

#section 4
python print_dither.py | tee dither_mats.log
for (( size=2; size<=8; size*=2 ))
do
python order_dithering.py "$img_dir"/house.tif $size | tee dither_"$size".log
done
mv ./*.tif output/sec4/
mv ./*.pdf output/sec4/
mv ./*.log output/sec4/
echo 'section 4 done'

#section 5
python error_diffusion.py "$img_dir"/house.tif | tee sec5.log
mv ./*.pdf output/sec5/
mv ./*.tif output/sec5/
mv ./*.log output/sec5
echo 'section 5 done'
