% Assessment 4 Q18 V07
clc; clear all;

%define problem
n=2;
f=@(x) 12*x(1)*x(2);
g1=@(x) 1 - (x(1)*x(2)^2)/22E6; % NL
g2=@(x) 1 - (x(1)*x(2))/25E4; % NL
g3=@(x) -7*x(1)+x(2); % Linear
g=@(x) [g1(x);g2(x);g3(x)];
x0=[445;245];
del_=0.9; % delta bar
%gradients
grad_f=@(x) [12*x(2);12*x(1)];
grad_g1=@(x) (-1/22E6)*[x(2)^2;2*x(1)*x(2)];
grad_g2=@(x) (-1/25E4)*[x(2);x(1)];
grad_g3=@(x) [-7;1];

% part A
% find s1 using quadprog
% g2 is violated using x0, 3n=6 ie include all constraints
delta=[del_;del_;1]; %only g3 was linear constraint

%ineqaulity constraint
A=[grad_g1(x0)';grad_g2(x0)';grad_g3(x0)'];
b=-delta.*g(x0);

%equality constraints
Aeq=[];
beq=[];

%bounds
LB=[];
UB=[];

% quadratic obj function parameters
B=eye(n,n);%H
[s,fval,exitflag,output,lambda]=quadprog(B,grad_f(x0),A,b,Aeq,beq,LB,UB);

%print answer
s % part A
lam=lambda.ineqlin % part B

% part C phi(a) at 
u=abs(lambda.ineqlin);
a=0;
phi_a0=phi(a,x0,s,u) % 3.0693e+06

% part D a_star for phi using fminbnd
a_lb=0;
a_ub=2;
% syms a;
% x1=x0+a*s;
% sym_phi=phi(f(x1),g(x1),[],u)

[a_star,phi,EXITFLAG,OUTPUT]= fminbnd('phi',a_lb,a_ub,[],x0,s,u);
a_star % 0.9310

% part E
% find updated B
x1=x0+a_star*s;
[f,grad_f]=fun(x1);
[g,h,grad_g,grad_h]=constraint_fun(x1);

p=a_star*s;
y=change_vector_y(x1,x0,lam);

%check find theta
if p'*y>=0.2*p'*B*p
    theta=1;
else
    theta=0.8*(p'*B*p)/(p'*B*p-p'*y);
end

%find n
n=theta*y+(1-theta)*B*p;

B_star=B-(B*p*p'*B)/(p'*B*p)+(n*n')/(p'*n)
%     0.7731   -0.4641
%    -0.4641    0.5373
