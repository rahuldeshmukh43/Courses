% this file provides input variables to the genetic algorithm
% upper and lower bounds, and number of bits chosen for "egg-crate" problem
% Modified on 11/10/09 by Bill Crossley.
clc;
close all;
clear all;

options = goptions([]);

vlb = [-2 -2];	%Lower bound of each gene - all variables
vub = [2 2];	%Upper bound of each gene - all variables
bits =[20    20];	%number of bits describing each gene - all variables


[x,fbest,stats,nfit,fgen,lgen,lfit]= GA550('GAfunc',[ ],options,vlb,vub,bits);

l=sum(bits); % length of chromosome
Npop=4*l     % total Population size
Mut_rate= (l+1)/(2*Npop*l) % probability for mutation
Ngen=size(stats,1)         % number of generations ran by GA
nfit                       % number of fitness evaluations
x                          % optimal design found by GA
fbest                      % fitness value for optimal solution

