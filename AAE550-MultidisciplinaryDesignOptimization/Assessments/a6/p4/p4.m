% assessment 6: Q25 V02: Goal Attainment

clc; clear all;

% function
f1= @(x) (x(1)+x(2)-7.5)^2 +0.45*(x(2)-x(1)+3)^2;
f2= @(x) 0.65*(x(1)-1)^2+0.5*(x(2)-4)^2;

% constraints
A=[];
b=[];
Aeq=[];
beq=[];

% bounds
UB=[];
LB=[0;0];

% 
a=[ 0.5,0.5;
    0.7,0.3;
    0.8,0.2;
    0.85,0.15;
    0.9,0.1];

x0=10*ones(2,1);
options = optimoptions('fmincon', 'Algorithm', 'sqp','Display','off');

fl_min=zeros(2,1);
for i=1:2
    [temp,fval,exitflag,output]=fmincon(eval(strcat('f',num2str(i))),x0,A,b,Aeq,beq,LB,UB,@(x)NonLcon(x),options);
    fl_min(i)=fval;
end
fl_min % utopia point
fprintf('----------------------------------------------');

% gamma conversion and teh noptimisation
gamma0=0;
x0=[x0;gamma0];
UB=[];
LB=[0;0;-inf];
f_g=fl_min;

for i =1: size(a,1)
    ia=a(i,:) 
    [x_star,gamma_star,exitflag,output]=fmincon(@(x) x(end),x0,A,b,Aeq,beq,LB,UB,...
                                        @(x)Goal_Att_cons(x,ia,f1,f2,f_g),...
                                        options);
    x_star(1:end-1); % actual x_star
    x_star(end); % gamma
    f_star=zeros(2,1); 
    f_star(1)=f1(x_star(1:end-1));
    f_star(2)=f2(x_star(1:end-1));
    f_star
    fprintf('----------------------------------------------');
end





