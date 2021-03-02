#!/bin/bash

##Section 2.1 and 2.2
#python3 ex2.py 1000 | tee sec2.log
#mv ./*.eps output/section_2/
#mv ./sec2.log output/section_2/

##section 4
#python3 ex4.py 
#mv ./*.pdf output/section4/

#section 5
for opt in {0..4}
do
	{
	echo option-"$opt";
	python3 ex5.py $opt; 
	echo "-----------------------";
	} | tee ex5_opt_"$opt".log
done
mv ./*.log output/section5/
