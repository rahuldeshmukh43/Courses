% constraints
function [g,h,grad_g,grad_h]=constraint_fun(x)

g=zeros(3,1);
% ineq constraint
g(1)= 1 - (x(1)*x(2)^2)/22E6; % NL
g(2)= 1 - (x(1)*x(2))/25E4; % NL
g(3)= -7*x(1)+x(2); % Linear

% eq constraints
h=[];

if nargout > 2 
    grad_g=[];
    grad_g=[grad_g,(-1/22E6)*[x(2)^2;2*x(1)*x(2)]];
    grad_g=[grad_g,(-1/25E4)*[x(2);x(1)]];
    grad_g=[grad_g,[-7;1]];
    
    grad_h=[];
    
end
