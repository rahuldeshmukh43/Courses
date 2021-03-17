#!/bin/bash

#ex2
python ex2.py ../../data.npy
mv ./*.pdf output/ex2
echo 'ex 2 done'

#ex3
python ex3.py ../../data.npy
mv ./*.pdf output/ex3
echo 'ex 3 done'

#ex4
name='d65'
python -W ignore ex4.py ../../data.npy ../../reflect.npy $name | tee "$name".log
#gamma correct rgb.tif
python -W ignore gamma_correction.py ./rgb_"$name".tif --linear -gout 2.2
mv ./*.tif output/ex4/
mv ./*.pdf output/ex4/
mv ./*.log output/ex4/
name='ee'
python -W ignore ex4.py ../../data.npy ../../reflect.npy $name | tee "$name".log
#gamma correct rgb.tif
python -W ignore gamma_correction.py ./rgb_"$name".tif --linear -gout 2.2
mv ./*.tif output/ex4/
mv ./*.pdf output/ex4/
mv ./*.log output/ex4/
echo 'ex 4 done'

#ex5
python -W ignore ex5.py ../../data.npy
mv ./*.pdf output/ex5/
echo 'ex5 done'
