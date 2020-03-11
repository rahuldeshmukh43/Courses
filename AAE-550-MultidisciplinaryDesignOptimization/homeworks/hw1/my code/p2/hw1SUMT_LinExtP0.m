function P = hw1SUMT_LinExtP0(x,tr_e)
% This function is the pseudo-objective function using Linear
% extended interior penalty method.
% input: x: col vector of design variables
%       tr_e: transition eps
% This does not include constraint scaling parameters, c_j.


% compute values of objective function and constraints at current x
f = hw1SUMTfun(x);
g = hw1SUMTcon(x);

% Linear extended Interior Penalty function
P=0;
for i=2:length(g)
    if g(i)<=tr_e
        g_hat=-1/g(i);
    else
        g_hat= - (2*tr_e-g(i))/tr_e^2;
    end
    P=P+g_hat;
end

end