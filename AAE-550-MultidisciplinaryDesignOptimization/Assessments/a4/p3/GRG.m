% Assessment 4 Q17 GRG 
clc; clear all;

f=@(x) x(1)^3+8*x(2)^2-7*x(1)*x(2);

h=@(x) x(1)^2+3*x(1)*x(2)+2;
g=@(x) 2*x(1)-x(2)^2/3-1+x(3); %x(3) is the slack variable

n=2; % x1,x2
m=1;
l=1;  %dep= m+l=2 indp=n-l=1

grad_f=@(x) [3*x(1)^2-7*x(2);16*x(2)-7*x(1);0];
grad_h1=@(x)[2*x(1)+3*x(2);3*x(1);0];
grad_h2=@(x)[2;-2*x(2)/3;1];

% part A
x1=0.29;
x2=-(2+x1^2)/(3*x1); % -2.3955
x3=-(2*x1-x2^2/3-1); % 2.3328

% Q=[grad_h1([x1,x2,x3])';grad_h2([x1,x2,x3])']
% % find independetn variables and dependent variable using [Q,I]
% T=[Q,eye(m+l,m+l)]
% % x1 is 1st dependent variable make canonical col
% T(1,:)=T(1,:)/T(1,1);
% T(2,:)=T(2,:)-T(2,1)*T(1,:)
% % x2 is the 2nd dependent variable 
% T(2,:)=T(2,:)/T(2,2);
% T(1,:)=T(1,:)-T(1,2)*T(2,:)
% y=[x1,x2]; z=[x3]
%A=[grad_h1([x1,x2,x3])(end);grad_h2([x1,x2,x3])(end)];%(m+l)x(n-l)=2,1 
%B=[grad_h1([x1,x2,x3])(1:2);grad_h2([x1,x2,x3])(1:2)];%(m+l)x(m+l)=2,2

g_f=grad_f([x1,x2,x3]);
g_h1=grad_h1([x1,x2,x3]);
g_h2=grad_h2([x1,x2,x3]);
all=[1,2,3];
indp=1;% Z

dep=all;
dep(indp)='';
%make  A and B using info of independent and dep
A=[g_h1(indp)';g_h2(indp)']
B=[g_h1(dep)';g_h2(dep)']

Gr=g_f(indp)-(inv(B)*A)'*g_f(dep)