%AAE:550 HW1 P2 
% Rahul Deshmukh
% PUID: 00 
%%
%  This file calls fminunc to minimize   
% This will use the linear extended interior penalty function; 
% note that there are no constraint scaling
% parameters (c_j) here; you may need these for your HW problem.

clc;clear all;
format long    % use format long to see differences near convergence

x0=[0.475058013560474;...
   13.235315387885350;...
   9.002524127606579];  % initial design
    
p = 0;         % initial value of minimization counter 
gamma= 5;      % Penalty multiplier
a=1/2;         % parameter for transition eps
tr_e= - 0.2;   % initial transition eps

% compute function value at x0, initialize convergence criteria
f = hw1SUMTfun(x0);
g = hw1SUMTcon(x0);
P0=hw1SUMT_LinExtP0(x0,tr_e);

r_p = abs(f)/P0;     % initial value of penalty multiplier
C = -tr_e/(r_p)^a;   % C constant parameter in tr_e equation

f_last = 2 * f;   % ensure that first loop does not trigger convergence
% set optimization options - use default BFGS with numerical gradients
% provide display each iteration of each minimization
options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');%, ...
%'Display', 'iter');  

obj_tol=1e-6; % relative tolerance for Objective function value
gj_tol=1e-6; % tolerance for ineqaulity constraint value
while ((abs((f-f_last)/f_last) >= obj_tol) || (max(g) >= gj_tol))
    f_last = f;  % store last objective function value
    p            % display current minimization counter
    r_p          % display current penalty multiplier
    % call fminunc - use "phi" pseudo-objective function, note that r_p is
    % passed as a "parameter", no semi-colon to display results
    [xstar,phistar,exitflag,output,grad,hessian] = fminunc(@hw1SUMTphi_LinExt,...
        x0,options,r_p,tr_e);
    % compute objective and constraints at current xstar
    x0
    xstar
    f = hw1SUMTfun(xstar)
    g = hw1SUMTcon(xstar)
    output.iterations
    exitflag
    
    p = p + 1;         % increment minimization counter
    r_p = r_p / gamma; % decrease penalty multiplier
    tr_e = -C*(r_p)^a; % update transition eps
    x0 = xstar;        % use current xstar as next x0
    fprintf('__________________________________________________________');
end
% display function and constraint values at last solution
f = hw1SUMTfun(xstar)
g = hw1SUMTcon(xstar)
%final solution
cr=1/xstar(1)
b=xstar(2)
a=xstar(3)
format short

%{
results

cr = 0.825867449871203
b =  13.999932794893834
a =  0.876182044674636

min drag = 99.974914369053806
x_star is feasible
%}


