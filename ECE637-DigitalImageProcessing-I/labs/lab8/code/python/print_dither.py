#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:15:10 2021

@author: rahul
"""
from order_dithering import Dither_mat

for i in [2,4,8]:
    d = Dither_mat(i);
    print('Bayer index matrix: size %dx%d'%(i,i))
    print(d)