function phi = hw1SUMTphi(x,r_p)
% This function is the pseudo-objective function using the exterior penalty.
% In this function, r_p is a "parameter", x are the variables.  This
% does not include constraint scaling parameters, c_j.

% compute values of the objective function and constraints at the current
% value of x
f = hw1SUMTfun(x);
g = hw1SUMTcon(x);

% exterior penalty function
P =max(0,g);  % note: no c_j scaling parameters
P=P'*P;

phi = f + r_p * P;
end