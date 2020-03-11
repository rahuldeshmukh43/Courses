%polynomial approximation
clc; clear all; close all;
format longE;
f = @(x) 2*x.^3+27*x.^2+3;
df=@(x) 6*x.^2+54*x;
x1=-3;
x2=0;
x3=6;
N=10;
tol=1e-6;
sol=zeros(N,1);
count=0;
notconverged=true;
while count<N && notconverged
    count=count+1;
    
    A=[1 x1 x1^2;1 x2 x2^2;1 x3 x3^2];
    b=[f(x1);f(x2);f(x3)];
    a=A\b;
    q=@(x) a(1)+a(2)*x+a(3)*x^2;
    dq=@(x) a(2)+2*a(3)*x;
    sol(count)=-a(2)/(2*a(3));
    %update
    temp=[f(x1),f(x2),f(x3),f(sol(count))];
    tempx=[x1,x2,x3,sol(count)];
    [maxterm,imax]=max(temp);
    tempx(imax)='';
    temp(imax)='';
    x1=tempx(1);
    x2=tempx(2);
    x3=tempx(3);
    
    if count>2
        if abs(sol(count)-sol(count-1))<tol
            notconverged=false;
        end
    end
end
sol
f(sol)
df(sol(count))

% A=[1 x1 x1^2;1 x2 x2^2;1 x3 x3^2];
% b=[f(x1);f(x2);f(x3)];
% a=A\b;
% q=@(x) a(1)+a(2)*x+a(3)*x^2;
% dq=@(x) a(2)+2*a(3)*x;
% plot([x1,x2,x3],[q(x1),q(x2),q(x3)])
