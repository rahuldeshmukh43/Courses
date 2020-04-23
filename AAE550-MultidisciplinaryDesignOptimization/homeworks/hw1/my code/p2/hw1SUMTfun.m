%AAE:550 HW1 P2 
% Rahul Deshmukh
% PUID: 0030004932
%%
function f = hw1SUMTfun(x)
% funciton computes the objective function value
% input:  x is a col vector
% output: f is a scalar value of the objective function

% x=[1/cr,b,a]
% objective function
y_f=0.033949298754935466466210448881266;%scalar multiplier
f = y_f*(x(2)^2)*(x(3)+3)^2;
end

