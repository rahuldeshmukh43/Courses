%AAE:550 HW1 P2 
% Rahul Deshmukh
% PUID: 0030004932
%%
function [f,gradF] = hw1funcwgrad(x)
% function and its analytical gradient
% input: x: is a col vetor
% output: f: scalar value of obj fn
%         gradF: col vector, analytical gradient of f


% x=[1/cr,b,a]
y_f=111.44897353114793675304812440719;%scalar multiplier
f = y_f*(x(2)^2)*(x(3)+3)^2;

%Find the gradient analytically  
if nargout > 1
	gradF = y_f*[0;
            2*x(2)*(x(3)+3)^2; ...
            2*(x(2)^2)*(x(3)+3)];
end
end

