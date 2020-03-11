%assessment 2 q9
%part A
%get gradient usingt syms
clc; clear all; close all;
% syms x1 x2;
% sym_f=17*(x1*x2-x1^2)^2+(5-3*x1)^2;
% sym_df=gradient(sym_f,[x1,x2])

f=@(x1,x2) 17*(x1*x2-x1^2)^2+(5-3*x1)^2;
df=@(x1,x2) [18*x1 - 34*(2*x1 - x2)*(- x1^2 + x2*x1) - 30;
    34*x1*(- x1^2 + x2*x1)];
xc=[3;-5];
df_xc= df(xc(1),xc(2));%[9000,-2448]

%part B: numerical derivative at xc with delta_x=1e-4
format long
delta_x=1e-4;
df_central=[(f(xc(1)+delta_x,xc(2))-f(xc(1)-delta_x,xc(2)))/(2*delta_x);
    (f(xc(1),xc(2)+delta_x)-f(xc(1),xc(2)-delta_x))/(2*delta_x)]; % 1.0e+03 * [ 9.000000000105501  -2.447999999913009]

%part C using fwd difference
delta_x=1e-4;
df_fwd1=[(f(xc(1)+delta_x,xc(2))-f(xc(1),xc(2)))/(delta_x);
    (f(xc(1),xc(2)+delta_x)-f(xc(1),xc(2)))/(delta_x)]; % 1.0e+03 [9.000288203751552  -2.447984699992958]

%part D using fwd difference
delta_x=1e-2;
df_fwd2=[(f(xc(1)+delta_x,xc(2))-f(xc(1),xc(2)))/(delta_x);
    (f(xc(1),xc(2)+delta_x)-f(xc(1),xc(2)))/(delta_x)];%   1.0e+03 *[9.028857416999745  -2.446470000000045]


%part E using fwd difference
delta_x=1e-6;
df_fwd3=[(f(xc(1)+delta_x,xc(2))-f(xc(1),xc(2)))/(delta_x);
    (f(xc(1),xc(2)+delta_x)-f(xc(1),xc(2)))/(delta_x)]%   1.0e+03 *[9.000002883112757 -2.447999846481252]