%HW1 problem (1)
%Rahul Deshmukh PUID: 00 
%deshmuk5@purdue.edu

% main file for problem 1
%---------------begin----------------------%
clc; clear all;close all;
format long;
%%
% (3)
% using BFGS
u0=[0;0];%initial guess: assuming at no displacement
%------------------Part A--------------------%
% using finite diffrence gradients and BFGS solver
options_3a=optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',false,...
    'Display','iter');
% [u_star,pe_star,exitflag,output,grad,hessian]=fminunc(@hw1_p1_PEfun,u0,options_3a)
%-----------results--------%
% u0=[0;0];u_star=[0.029448184643855;0.044483086047248];
% f(u_star)= -3.733027694338580e+02;
% grad=  1.0e-05*[0;-0.762939453125000]; num_iter=3; funcCount=18; exitflag=1;
%---------------------------%

%-----------------Part B------------------------%
%  solve using analytic gradients 
options_3b=optimoptions(@fminunc,'Algorithm','quasi-newton',...
    'SpecifyObjectiveGradient', true, 'Display', 'iter');
% [u_star,pe_star,exitflag,output,grad,hessian]=fminunc(@hw1_p1_PEwtgrad,u0,options_3b)
%-----------results--------%
% u0=[0;0];u_star=[0.029448192103366;0.044483093518566];
% f(u_star)=  -3.733027694338724e+02;
% grad=[0;0]; num_iter=3; funcCount=6; exitflag=1;
%---------------------------%

%%
% (4)
% using DFP and Steepest Descent
u0=[0;0];%initial guess: assuming at no displacement
%------------------Part A--------------------%
% using analytic gradient with DFP update
options_4a=optimoptions(@fminunc,'Algorithm','quasi-newton','SpecifyObjectiveGradient',...
    true,'Display','iter','HessUpdate','dfp');
% [u_star,pe_star,exitflag,output,grad,hessian]=fminunc(@hw1_p1_PEwtgrad,u0,options_4a)
%-----------results--------%
% u0=[0;0];u_star=[0.029448192103366;0.044483093518566];
% f(u_star)=  -3.733027694338725e+02;
% grad=[0;0]; num_iter=3; funcCount=6; exitflag=1;
%---------------------------%

%-----------------Part B------------------------%
%  solve using analytic gradients with Steepest Descent
options_4b=optimoptions(@fminunc,'Algorithm','quasi-newton','SpecifyObjectiveGradient',...
    true,'Display','iter','HessUpdate','steepdesc');
% [u_star,pe_star,exitflag,output,grad,hessian]=fminunc(@hw1_p1_PEwtgrad,u0,options_4b)
%-----------results--------%
% u0=[0;0];u_star3a=[0.029448178767296;0.044483073374022];
% f(u_star3a)=  -3.733027694337959e+02;
% grad=[-0.003359750103300;-0.005376640310715]; num_iter=4; funcCount=25; exitflag=1;
%---------------------------%

%%
% (5)
% using newtons method
u0=[0;0];%initial guess: assuming at no displacement

% using newtons mehod with specified hessian  and gradient
options_5=optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',...
    true,'Display','iter','HessianFcn','objective');
% [u_star,pe_star,exitflag,output,grad,hessian]=fminunc(@hw1_p1_HessianPEfun,u0,options_5)
%-----------results--------%
% u0=[0;0];u_star=[0.029448192103366;0.044483093518566];
% f(u_star)=  -3.733027694338725e+02;
% grad= 1.0e-11*[-0.090949470177293;-0.363797880709171]; num_iter=1; funcCount=2; exitflag=1;
%---------------------------%

%%
% (8)
% solving for u using Ku=p
[K,p]=hw1_p1_tangent();
u_star=K\p
%-----------results------------%
% u_star=[0.029448192103366;0.044483093518566];
% in one step, implicit system.
%-------------------------------%

























