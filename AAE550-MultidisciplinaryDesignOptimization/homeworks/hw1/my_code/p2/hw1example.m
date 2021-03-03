% This file calls fminunc to minimize the example function from the AAE 550
% class notes  - see slide 23, Class 07.
% Prof. Crossley 5 Sep 2018
clear all
% use long format to see difference in solution
format long
% This version uses numerical derivatives
x0 = [-2; 4];   % initial design
options = optimoptions(@fminunc, 'Algorithm','quasi-newton', ...
    'Display', 'iter');  % display each iteration
[xnum,fval,exitflag,output,grad] = fminunc(@hw1func,x0,options)  % lots of information 

%This version uses analytic derivatives
x0 = [-2; 4];   % initial design

% Legacy optimoptions for optimset
options = optimoptions(@fminunc, 'Algorithm','quasi-newton', ...
    'GradObj', 'on', 'Display','iter');

% Current optimoptions for optimset
% options = optimoptions(@fminunc, 'Algorithm','quasi-newton', ...
%     'SpecifyObjectiveGradient', true, 'Display','iter');

[xana,fval,exitflag,output,grada] = fminunc(@hw1funcwgrad,x0,options)  % lots of information 

% return to short format
format short