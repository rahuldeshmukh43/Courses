% this file provides input for and calls fminsearch to use the Nelder-Mead
% Simplex method
% Modified on 11/05/07 by Bill Crossley.

close all;
clear all;

options = optimset('Display','iter');

x0 = [-pi; 3*pi/2];

[x,fval,exitflag,output] = fminsearch('NMfunc',x0,options)

