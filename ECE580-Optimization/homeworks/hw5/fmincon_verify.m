% ECE 580 HW5: Problem 5,6,7 verfication using fmincon
% Rahul Deshmukh 
% deshmuk5@purdue.edu
clc; clear all;
%% P5: have the correct solution
fprintf('\n-------P5------------\n')
f = @(x) -(4*x(1)+x(2)^2);
A=[];
b=[];
Aeq=[];
beq=[];
LB=[];
UB=[];
X0=[2,-1];
[x_str,fval] = fmincon(f,X0,A,b,Aeq,beq,LB,UB,@p5con)

%% P6: have the correct solution
fprintf('\n-------P6------------\n')
f = @(x) +(18*x(1)^2 -8*x(1)*x(2) +12*x(2)^2);
A=[];
b=[];
Aeq=[];
beq=[];
LB=[];
UB=[];
X0=[0,0];
[x_str,fval] = fmincon(f,X0,A,b,Aeq,beq,LB,UB,@p6con)

%% P7a: have the same solution but the extremizer is not a strict minimizer
fprintf('\n-------P7a------------\n')
f = @(x) (x(1)^2 +x(2)^2 -2*x(1) -10*x(2) +26);
A=[5,1/2];
b=[5];
Aeq=[];
beq=[];
LB=[];
UB=[];
% X0=[-48+10*sqrt(23),23020-4800*sqrt(23)];
X0 =[0,0];
[x_str,fval] = fmincon(f,X0,A,b,Aeq,beq,LB,UB,@p7acon)

% plot function and constraints
a = 10;
fp = @(x_1,x_2) (x_1.^2 +x_2.^2 -2*x_1 -10*x_2 +26);
x = -a:0.1:a;
[X,Y] = meshgrid(x,x);
Z = fp(X,Y);
g1 = 5*x.^2;
g2 = 10-10*x;
fig = figure(1);
hold on;
contour(X,Y,Z);
plot(x,g1,'k');
plot(x,g2,'r');
plot(x_str(1),x_str(2),'rx','MarkerSize',10);
plot(-48+10*sqrt(23),23020-4800*sqrt(23),'bx','MarkerSize',10);
xlim([-a,a]);
ylim([-a,a]);
xlabel('x_1');
ylabel('x_2');
saveas(fig,'./pix/p7a_plot.epsc')
%% P7b: have the correct answer
fprintf('\n-------P7b------------\n')
f = @(x) x(1)^2 + x(2)^2;
A=[-1,-1];
b=[-5];
Aeq=[];
beq=[];
LB=[0;0];
UB=[];
X0=[0,0];
[x_str,fval] = fmincon(f,X0,A,b,Aeq,beq,LB,UB)

%% P7c: have the correct solution
fprintf('\n-------P7c------------\n')
f = @(x) x(1)^2 + 6*x(1)*x(2) -4*x(1) -2*x(2);
A=[2,-2];
b=[1];
Aeq=[];
beq=[];
LB=[];
UB=[];
X0=[1,1];
[x_str,fval] = fmincon(f,X0,A,b,Aeq,beq,LB,UB,@p7ccon)

%%
function [C,Ceq]=p5con(x)
    C=[];
    Ceq= x(1)^2 + x(2)^2 -9;
end

function [C,Ceq]=p6con(x)
    C=[];
    Ceq= 1-(2*x(1)^2+ 2*x(2)^2);
end

function [C,Ceq]=p7acon(x)
    C=x(2)/5-x(1)^2;    
    Ceq=[];    
end

function [C,Ceq]=p7ccon(x)
    C=x(1)^2 + 2*x(2) -1;    
    Ceq=[];    
end