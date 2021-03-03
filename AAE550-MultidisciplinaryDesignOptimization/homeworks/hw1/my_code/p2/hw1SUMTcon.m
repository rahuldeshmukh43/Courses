%AAE:550 HW1 P2 
% Rahul Deshmukh
% PUID: 00 
%%
function g = hw1SUMTcon(x)
% this function computes the inequality constraint function values
% input: x is a column vector
% output: g: col vector of constraint values [bounds; gj]

% x=[x1,x2,x3]=[1/cr,b,a]
% constraints
g=zeros(9,1);%col vector
% bounds
g(1)=1-3*x(1);%for cr
g(2)=0.8*x(1)-1;

g(3)=x(2)/14-1;%for b
g(4)=1-x(2)/8;

g(5)=x(3)/10-1;%for alpha
g(6)= -1 - x(3)/5;

% inequality constraints
% for lift coeff
y_lc=0.01096622711232150957648276777764;
g(7)=y_lc*x(2)*(x(3)+3)*x(1)/0.9-1;
g(8)=1-(y_lc*x(2)*(x(3)+3)*x(1))/0.7;

% for Lift
y_L=12.50454558912777413506277959447;
g(9)=1-y_L*(x(2)^2)*(x(3)+3)/9500;
end