% this file provides input variables and parameters for simulated annealing
% upper and lower bounds, and initial design chosen for "egg-crate" problem
% Modified on 11/12/02 by Bill Crossley.
clc;
close all;
clear all;

bounds = [-2 2;
          -2 2];        % upper and lower bounds for each of the two variables

X0 = [2; 2];            % initial design NOTE: this is a column vector

options = zeros(1,9);	% set up options array for non-default inputs

options(1) = 28;		% initial temperature (default = 50) 
options(6) = 0.6;		% cooling rate (default = 0.5)       

[xstar,fstar,count,accept,oob]=SA_550('SAfunc',bounds,X0,options);

