% Assessment 7: Q28 Response Surface Application V04- last was wrong, V01
clc; clear all;

% load given data
data=load('ques29v4.dat');
% assign values to variables
A1= data(:,1);
A2= data(:,2);
A3= data(:,3);
m = data(:,4);
s1= data(:,5);
s2= data(:,6);
s3= data(:,7);

sa=75300; %psi % v4
% sa= 25000; % psi v1
% make the basis matrix X
n=3; % number of design variables
X = [A1,A2,A3];
combinations = combnk(1:n,2);
XiXj = [];
for i=1:size(combinations,1)
    c1= combinations(i,1);
    c2= combinations(i,2);
    XiXj = [XiXj, X(:,c1).*X(:,c2)];
end

X = [ones(length(A1),1), X, XiXj, X.^2];
% Part A: 3
size(X)
% find coefficients for m, s1,s2,s3
% Part B: 4
a_m  = X\m
% Part C: 1
a_s1 = X\s1
a_s2 = X\s2
a_s3 = X\s3

% Part D: ?
% fmincon for finding optimised solution to the approximate problem
a_g = [a_s1,a_s2,-1*a_s3]/sa

A=[]; % no linear constraint all quad cons
b=[];
Aeq = []; % no equality constraint
beq=[];
LB= 0.1*ones(n,1);
UB= 5.0*ones(n,1);

options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
% options = optimoptions('fmincon','Algorithm','sqp','Display','iter',...
%     'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
%              'DerivativeCheck','on');

x0=(LB+UB)/2.0;
% x0 = [2.5688;1.6030;0.1000];
% x0 = UB;
[x_star,fval,exitflag,output] = fmincon(@(x)fun(x,a_m),x0,A,b,Aeq,beq,...
                                         LB,UB,@(x)NonLinCon(x,a_g),options);
    
x_star
fval
exitflag
