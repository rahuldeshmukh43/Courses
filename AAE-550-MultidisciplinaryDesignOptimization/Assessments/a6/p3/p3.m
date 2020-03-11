% Assessment 6 Q24 V08: Min-Max Approach
clc; clear all;
format short;
% functions
f1=@(x) x(1)^2-1.25*x(2)^2;
f2= @(x) 2*x(1)/x(2);

% constraints
A=[];
b=[];

Aeq=[];
beq=[];

% bounds
LB=[-1;1];
UB=[2;2];

x0=[1;1];
ia1=0:0.25:1;
ia2=1:-0.25:0;
a=[ia1;ia2];

% fmincon for individual fi subject to constraints and bounds
options = optimoptions('fmincon', 'Algorithm', 'sqp','Display','off');
fl_min=zeros(2,1);
for i=1:2
    [temp,fval,exitflag,output]=fmincon(eval(strcat('f',num2str(i))),x0,A,b,Aeq,beq,LB,UB,[],options);
    fl_min(i)=fval;
end
fl_min % utopia point
fprintf('----------------------------------------------');
% beta conversion and teh noptimisation
beta0=100;
x0=[x0;beta0];
LB=[-1;1;-inf];
UB=[2;2;inf];
for i =1: size(a,2)
    ia=a(:,i)   
    
    [x_star,beta_star,exitflag,output]=fmincon(@(x) x(end),x0,A,b,Aeq,beq,LB,UB,...
                                        @(x)NonLcon(x,a(:,i),fl_min,f1,f2),...
                                        options);
    x_star(1:end-1); % actual x_star
    x_star(end); % beta
    f_star=zeros(2,1); 
    f_star(1)=f1(x_star(1:end-1));
    f_star(2)=f2(x_star(1:end-1));
    f_star
    fprintf('----------------------------------------------');
end

