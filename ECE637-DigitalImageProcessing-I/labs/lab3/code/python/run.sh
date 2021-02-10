#!/bin/bash
program=view_segmentatation.py
path=../C-code/run/output

for t in 2 1 3
do
	python "$program" "$path"/segmented_"$t".tif
done

mv ./*.eps ../../pix/

echo 'done'
