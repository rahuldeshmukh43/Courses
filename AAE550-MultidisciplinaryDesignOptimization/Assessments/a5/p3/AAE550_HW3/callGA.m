% this file provides input variables to the genetic algorithm
% upper and lower bounds, and number of bits chosen for "egg-crate" problem
% Modified on 11/10/09 by Bill Crossley.

close all;
clear all;

options = goptions([]);

vlb = [-2*pi -2*pi];	%Lower bound of each gene - all variables
vub = [2*pi 2*pi];	%Upper bound of each gene - all variables
bits =[10    10];	%number of bits describing each gene - all variables


[x,fbest,stats,nfit,fgen,lgen,lfit]= GA550('GAfunc',[ ],options,vlb,vub,bits);


