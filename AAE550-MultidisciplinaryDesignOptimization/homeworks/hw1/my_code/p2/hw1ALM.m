%AAE:550 HW1 P2 
% Rahul Deshmukh
% PUID: 00 
%%
% This file calls fminunc to minimize 
% This will use the Augmented Lagrangian Function.  
% No constraint scaling parameters (c_j) used here.
% This is essentially the same format as the exterior penalty example
% , with the exception of the lambda update and the 'hw1SUMTalm' function.

clc;
clear all
format long    % use format long to see differences near convergence

% x0 = [0;0;0];  % initial design
x0=[   0.350113005416948;   8.217596212161899;  -2.530475785010287]; 
p = 0;         % initial value of minimization counter 
r_p = 1.0;     % initial value of penalty multiplier 
lambda = zeros(9,1);     % initial Lagrange multipliers
y= 5;          % gamma

% compute function and constraint value at x0 for initialization ...
% of convergence criteria
f = hw1SUMTfun(x0);
g = hw1SUMTcon(x0);
f_last = 2 * f;   % ensure that first loop does not trigger convergence


% set optimization options - use default BFGS with numerical gradients
% provide display each iteration
options = optimoptions(@fminunc,'Algorithm', 'quasi-newton');%;, ...
 %'Display', 'iter');

obj_tol=1e-6; % relative tolerance for Objective function value
gj_tol=1e-6; % tolerance for ineqaulity constraint value
% hk_tol=1e-6; %tolerance for eqaulity constraint value
while ((abs((f-f_last)/f_last) >= obj_tol) || (max(g) >= gj_tol)) %|| (max(h)>=1e-5)
    f_last = f;  % store last objective function value
    p            % display current minimization counter
    lambda       % display current Lagrange multiplier values
    r_p          % display current penalty multiplier
    % call fminunc - use "ALM" pseudo-objective function, note that r_p and
    % lambda are passed as parameters, no semi-colon to display results
    [xstar,phistar,exitflag,output,grad,hessian] = fminunc(@hw1SUMTalm,...
        x0,options,r_p,lambda);
    %[xstar,phistar] = fminunc(@hw1SUMTalm,x0,options_2,r_p,lambda)
    % compute objective and constraints at current xstar
    x0
    xstar
    f = hw1SUMTfun(xstar)
    g = hw1SUMTcon(xstar)
    output.iterations
    exitflag
    
    len_g=length(g);
    % h = hw1SUMTcon_heq(xstar);
    % update lagrange multipliers
    lambda(1:len_g) = lambda(1:len_g) +...
        2 * r_p * max(g, -lambda(1:len_g)/(2*r_p));
    % lambda(len_g+1:end)=lambda(len_g+1:end)+2*r_p*h;
    p = p + 1;     % increment minimization counter
    r_p = r_p *y;  % increase penalty multiplier
    x0 = xstar;    % use current xstar as next x0
    fprintf('___________________________________________________________');
end
% display function and constraint values at last solution
f = hw1SUMTfun(xstar)
g = hw1SUMTcon(xstar)
%final solution
cr=1/xstar(1)
b=xstar(2)
a=xstar(3)
% h = hw1SUMTcon_heq(xstar)
% format short

%--------------results-------------------------%
% Numerical gradient
% drag = 99.973786369114805
% p=10
% gradient=   1.0e+05 *[0;   0.002007212006538;  0.002429008483887]
%  hessian =   1.0e+13 *[-0.000000000006226,-0.000000000000550,-0.000000000007706; 
% -0.000000000000550,0.216829627438697,0.209057500381814;
% -0.000000000007706, 0.209057500381814,2.417572089118957];
% xstar is feasible
% cr = 0.807996737452328
% b =  13.999999659075149
% a =  0.876141664870233
%----------------------------------------------%