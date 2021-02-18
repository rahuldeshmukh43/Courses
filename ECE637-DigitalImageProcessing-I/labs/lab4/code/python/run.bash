#!/bin/bash

#section 1
python histogram.py ../../kids.tif
python histogram.py ../../race.tif
mv ./*.eps output/section1/

#section 2
python hist_eq.py ../../kids.tif
python histogram.py kids_hist_eq.tif
mv ./*.eps output/section2/

#section 3
python contrast_stretch.py ../../kids.tif 80 160
python histogram.py kids_cont_st.tif
mv ./*.eps output/section3/

#section 4.2
#gamma=$1
gamma=170
python gamma_monitor_4.2.py $gamma
mv ./*.eps output/section4/

#section 4.3
python gamma_correction.py ../../linear.tif --linear
#python gamma_correction.py ../../gamma15.tif -gin 1.5 -gout 2.5
python gamma_correction.py ../../gamma15.tif -gin 1.5
mv ./*.eps output/section4/

echo "done"
