% assessment 4 Q16
clc;
clear all;

%define function 
n=3;
f=@(x) (x(1)+2)^2+(x(2)-4)^2+(x(3)+6)^2;
grad_f = @(x) [2*(x(1)+2);2*(x(2)-4);2*(x(3)+6)];

g=@(x) x(1)^2+x(2)^2+x(3)^2-11;
grad_g=@(x) [2*x(1);2*x(2);2*x(3)];

x0=[sqrt(3);2;2];
s0=[-1;1;-1];
t0=1;
e=-0.1;

%part A
% find usability and feasibility
grad_f(x0)'*s0 % -27.4641
grad_g(x0)'*s0 % -3.4641
% useable and feasible direction

%part B
% formulate the Zountendijk's search problem
g(x0)>e % constraint is found to be active and is a non linear constraint
J=1;
t1=(1-(g(x0))/e)*t0
A=[grad_g(x0)',t1;grad_f(x0)',1]
% A = 3.4641    4.0000    4.0000    1.0000
%     7.4641   -4.0000   16.0000    1.0000
p=[0;0;0;1];

% part C
I=eye(J+1,J+1);
B=-A*A';
c=-A*p;
% solve [B I]{u,v} = c using modified simplex
T=[B,I,c] % tableau
ratio=[];
for i=1:length(c)
    ratio=[ratio;T(i,end)/T(i,i)];
end
ratio % u1 enters v1 leaves
% row operations to make u1 column as canonical
T(1,:)=T(1,:)/T(1,1);
T(2,:)=T(2,:)-T(2,1)*T(1,:);
%print c
T(:,end) % vi was greater than zero v2= 0.6635 v1=0
T
u=[T(1,end);0] % [0.0222 0]
%solve for y and then s 
y=p-A'*u;
s=y(1:n) %   -0.0770   -0.0889   -0.0889

