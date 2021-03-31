#!/bin/bash

#section 1
python section1.py ../../img14g.tif ../../img14bl.tif | tee bl.log
python section1.py ../../img14g.tif ../../img14gn.tif | tee gn.log
python section1.py ../../img14g.tif ../../img14sp.tif | tee sp.log
mv ./*.pdf output/sec1/
#mv ./*.tif output/sec1/
mv ./*.log output/sec1/
echo 'section 1 done'
