clc; clear all; close all;
% syms x;
% f=12.7-3.88*x-(sqrt(7))*x^2+3*x^4;
% df= diff(f)
% ddf = diff(df)
% x1=x0-subs(df,x0)/subs(ddf,x0)
f=@(x) 12.7-3.88*x-(sqrt(7))*x.^2+3*x.^4;
df=@(x) 12*x.^3 - 2*7^(1/2).*x - 97/25;
ddf=@(x) 36*x.^2 - 2*7^(1/2);
x0=3;
N=10;
sol=zeros(N+1,1);
sol(1)=x0;
for i=1:N
    sol(i+1)=sol(i)-df(sol(i))/ddf(sol(i));
end
sol
f(sol)
% x1=x0-df(x0)/ddf(x0);
% x2=x1-df(x1)/ddf(x1);



