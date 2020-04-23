function phi = hw1SUMTphi_Int(x,r_p)
% This function is the pseudo-objective function using interior
% penalty method.
% input: x: col vector of design variables
%       r_p: penalty multiplier, x are the variables.  This
% does not include constraint scaling parameters, c_j.


% compute values of objective function and constraints at current x
f = hw1SUMTfun(x);
g = hw1SUMTcon(x);

% Interior penalty function
% classic
% P=sum(-1./g);

% Log barrier
P=sum(-log(-g));

phi = f + r_p * P;
end