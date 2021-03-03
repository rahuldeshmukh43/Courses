% AAE:550 HW1 P2 
% Rahul Deshmukh
% PUID: 00 
%%
function A = hw1SUMTalm(x,r_p,lambda)
% This function is the pseudo-objective function using the ALM.
%Input: x: column vector of design variables
%       r_p: penalty multiplier
%       lambda: col vector, lagrange multipliers[ineq, eq]

% compute values of the objective function and constraints at the current
% value of x
f = hw1SUMTfun(x);
g = hw1SUMTcon(x); %inequality constraints column vector of size len_g
% h = hw1SUMTcon_heq(x);% equaltiy constraints column vector of size len_h

len_g=length(g);
% Fletcher's substitution only for ineqaluti constraints
psi = max(g, -lambda(1:len_g) / (2 * r_p)); % col vec size len_g

% Augmented Lagrangian function
A = f + (lambda(1:len_g))' * psi + r_p * ((psi')*psi);%+...
%     (lambda(len_g+1:end))'*h+r_p*(h'*h);
end