%  Assessment 6 Q23 V06
% gaming approach
clc; clear all;

% function
f1=@(x) x(1)+9;
f2=@(x) exp(x(2))+6;
% constraints
A=[-2 -3;
    2  3];
b=[-6;12];

Aeq=[];
beq=[];
% bounds
LB=zeros(2,1);
UB=[inf;2];

x0=[1;1];
e=[7 9 12 13.5 10 9.5];

options = optimoptions('fmincon', 'Algorithm', 'sqp','Display','off');

%  Part A: f1(x*) with f2(x*)<=e1

for i=1:3
    ie=e(i);
    [x_star,f1_star,exitflag1,output1]=fmincon(f1,x0,A,b,Aeq,beq,LB,UB,@(x)nonLcon_f2(x,ie),options);
    f1_star
end
fprintf('-----------------------------');
for i=1:3
    ie=e(i+3);
    [x_star,f2_star,exitflag2,output2]=fmincon(f2,x0,A,b,Aeq,beq,LB,UB,@(x)nonLcon_f1(x,ie),options);
    f2_star
end
    