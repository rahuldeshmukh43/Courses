#!/bin/bash
program=../bin/ConnectedSet
#s0row=1
#s0col=1

s0row=45
s0col=67

for t in 2 1 3
do
	echo "T = "  $t
	$program ../../../img22gd2.tif  $s0row $s0col $t
	mv connected_set.tif ./output/cs_"$t".tif
	mv segmented.tif output/segmented_"$t".tif
done

echo 'done'

