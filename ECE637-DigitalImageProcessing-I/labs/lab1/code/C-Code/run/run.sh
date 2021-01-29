#!/bin/bash

LAM=1.5

../bin/ImageReadWrite_FIR ../../../img03.tif
mv color.tif output/img03_FIR.tif

../bin/ImageReadWrite_FIR_sharpen ../../../imgblur.tif $LAM
mv color.tif output/imgblur_FIR_sharp_"$LAM".tif

../bin/ImageReadWrite_FIR_sharpen ../../../imgblur.tif 0.8
mv color.tif output/imgblur_FIR_sharp_0.8.tif

../bin/ImageReadWrite_FIR_sharpen ../../../imgblur.tif 0.2
mv color.tif output/imgblur_FIR_sharp_0.2.tif

../bin/ImageReadWrite_IIR ../../../img03.tif
mv color.tif output/img03_IIR.tif

#convert tif to eps for latex
for image in $(ls ./output/*.tif)
do
	echo $image
	python ../../python/tiff2eps.py $image ./output/
done

echo "done"
