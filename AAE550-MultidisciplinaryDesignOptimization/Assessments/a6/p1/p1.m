% Assessment 6: Q22 Weighted Sum approach V03

clc; clear all;

%  function 
f1 = @(x) (x(1)-4)^2+(x(2)-9)^2;
f2 = @(x) (x(1)-9)^2+(x(2)-17)^2;

% constraints

A=[-5 -6;
    -3 4];
b=[-10;10];
Aeq=[];
beq=[];
% bounds
UB=[];
LB=[0.0;0.0];

x0=[10;10];

a=[0 0.01 0.5 0.8 1;
   1 0.99 0.5 0.2 0];

for u=1:size(a,2)
    % Part A: utopia point f0
    fl_min=zeros(2,1);
    
    % fmincon for individual fi subject to constraints and bounds
    options = optimoptions('fmincon', 'Algorithm', 'sqp');
    for i=1:2
        [temp,fval,exitflag,output]=fmincon(eval(strcat('f',num2str(i))),x0,A,b,Aeq,beq,LB,UB,@(x)NonLcon(x),options);
        fl_min(i)=fval;
    end
    fl_min % utopia point
    
    % Part B: wieghted sum
    % phi=zeros(2,1);
    % for i=1:2
    %     f_l=eval(strcat('f',num2str(i)));
    %    phi(i)= (( f_l-fl_min(i) )/ fl_min(i))^2;
    % end
    % phi
    
    [x_star,phi_star,exitflag2,output2]=fmincon(@(x)phi(x,a(:,u),fl_min,f1,f2),x0,A,b,Aeq,beq,LB,UB,@(x)NonLcon(x),options);
    x_star
    phi_star
    
    f_star=zeros(2,1);
    f_star(1)=f1(x_star); f_star(2)=f2(x_star); 
    f_star
    
%     x0=x_star;
    
end