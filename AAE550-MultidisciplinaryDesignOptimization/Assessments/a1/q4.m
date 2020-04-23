clc; clear all;
fx = @(x1,x2) 3*x1^2-2*x2^2;
g1= @(x1,x2) 2*x1^2+x2-18;
g2=@(x1,x2) -x1+x2+3;
dg1=@(x1,x2)[4*x1;1];
dg2=@(x1,x2)[-1;1];
df = @(x1,x2) [6*x1;-4*x2];
xc= [-3.5,-6.5];
%part A
b=df(xc(1),xc(2))
%part B
g1(xc(1),xc(2))%=0
g2(xc(1),xc(2))%=0
%both are zeors ie Lam1 and Lam2 are both unknowns
col1=dg1(xc(1),xc(2));
col2=dg2(xc(1),xc(2));
A=[col1,col2];
Lam=A\(-b)% [0.3846;-26.3846]
% Lam[2] <0 therefore KKT is not satisfied

