#!/bin/bash
#run file for C-code
program=../bin/MedianFiltering
img_dir=../../..
cd ../src/
make all
cd ../run/
$program "$img_dir"/img14gn.tif ./output/img14gn_medianfilter.tif
$program "$img_dir"/img14sp.tif ./output/img14sp_medianfilter.tif
convert output/img14gn_medianfilter.tif output/img14gn_medianfilter.png
convert output/img14sp_medianfilter.tif output/img14sp_medianfilter.png
echo 'section 2 done'
