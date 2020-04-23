% objective function
function [f,grad_f]=fun(x)
%  Input: x is the design vector column

f= 12*x(1)*x(2);

if nargout > 1  % fun called with two output arguments
    % Matlab naming convention will use grad_f(1) as df/dx(1); grad_f(2) as
    % df/dx(2)
    grad_f = [12*x(2);12*x(1)];

end