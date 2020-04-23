% assessment 2 part-1
%q-1
clc; clear all; close all;
phi=@(x) 7*x^5+8*x^2-2*x-3;
xi=@(x) 2*x^2-2*sin(2*x+1);
eta = @(x1,x2) 5*(x2-x1^2)^2+(1-x1)^2;
a=[2;-1];
s=[0.9752;0.0825];
%part A use fminbnd on phi 
tol=1e-6;
phixmin=-4;
phixmax=-1;
opt=optimset('TolX',tol);
[phix,phival,phiflag,phiout]=fminbnd(phi,phixmin,phixmax,opt); %phix=-4; phival=-7.0346e3; phiflag=1

%partB use fminbnd on xi 
[xix,xival,xiflag,xiout]=fminbnd(xi,phixmin,phixmax,opt);%xix=-1; xival=3.6832; xiflag=1

%part C
syms al; %alpha
temp=a+al*s;
x1=temp(1);
x2=temp(2);
vpa(expand(eta(x1,x2))); %display it
eta_al=@(al) 4.522148031531008*al^4 + 36.31260727232*al^3 + 121.39884149*al^2 + 192.8654*al + 126.0;

%partC use fminbnd to min eta_al 
al_min=0;
al_max=5;

[eta_al_al,eta_al_val,eta_alflag,eta_alout]=fminbnd(eta_al,al_min,al_max,opt);%eta_al_al=4.7455e-7; et_al_val=126.0001; eta_alflag=1