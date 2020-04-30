% ECE 580 HW5: Problem 3
% Rahul Deshmukh 
% deshmuk5@purdue.edu
clc; clear all; 
format rat;
%% include paths
addpath('../OptimModule/optimizers/linprog/');
%%
verbose=0;
c = [4; 3];
A = [-5, -1;
     -2, -1;
     -1, -2];
b = [-11; -8; -7];
Aeq = [];
beq = [];
LB=[];
UB=[];
[x_str, fval] = mylinprog(c,A,b,Aeq,beq,LB,UB,verbose)
