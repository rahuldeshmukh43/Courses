#!/bin/bash     

#Task1
#SpecAnal       
python3 MySpecAnal.py img04g.tif 0
#BetterSpecAnal                                                             
python3 MySpecAnal.py img04g.tif 1

#Task2
#generate filtered image
python3 generate_filtered_image.py filtered_image.tif
#mesh plot of log S_y
python3 MySpecAnal.py filtered_image.tif 0
#mesh plot of betterspecAnal
python3 MySpecAnal.py filtered_image.tif 1
