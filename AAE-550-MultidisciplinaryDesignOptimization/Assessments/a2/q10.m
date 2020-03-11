%assessment 2 q10
clc; clear all; close all;
format long;
f=@(x)  2*x(1)^2 + 3*x(2)^2 + x(3)^2 + 4*x(1)*x(2) + 0.2*x(2)*x(3);
x0=[5;10;1];

delta_x=1e-6;

%part A s1 using steepest desecent
s1=-1*df_fwd(f,x0,delta_x);%[-60,-80.2,-4]

%part B alpha for s1
%syms a; %alpha
%sym_x1=x0+a*s1; %update x
%find function f(a)
%sym_f1_a=vpa(expand(f(sym_x1))); %to show the function

% f1_a=@(a) 45824.283335688838938188477132308*a^2 - 10048.040367269095440860837697983*a + 553.0;
%minimize f1_a 
%sym_df1_a=diff(sym_f1_a)%
% df1_a=@(a) 91648.566671377677876376954264617*a - 10048.040367269095440860837697983; 
%find a using df1_a=0;
a1=0.109636634070865;

%part C
x1=x0+a1*s1;%update x

s2=-1*df_fwd(f,x1,delta_x);%   [1.484224595671435  -1.042350305624495  -1.364336029485713]

%part D
% syms a; %alpha
% sym_x2=x1+a*s2; %update x
% find function f(a)
% sym_f2_a=vpa(expand(f(sym_x2))); %to show the function

% f2_a= @(a) 3.6228359528590824738364286967167*a^2 - 5.1508280874150341979980835888348*a + 2.1833375622198219281155014680944;
% df2_a=@(a) 7.2456719057181649476728573934335*a - 5.1508280874150341979980835888348;

a2=0.710883428678310;
x2=x1+a2*s2;
s3=-1*df_fwd(f,x2,delta_x) %    [0.227740153513345  -0.622878855160991   0.723629632493061]