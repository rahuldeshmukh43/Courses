%AAE:550 HW1 P2 
% Rahul Deshmukh
% PUID: 0030004932
%%
%script to find x0 for exterior penalty function
% x=[1/cr,b,a]
clc; clear all;
format long;
notfound=1;
Nmax=10^4;%maximum number of iterations for the while loop
i=0;%counter
while notfound && i<=Nmax
    % x = a+ (b-a)*x_rand;
    %pick random value for cr,b,a conforming their bounds
    x1= 0.8 + (3-0.8)*rand(1);%cr
    x2= 8 + (14-8)*rand(1);%b
    x3= -5 + (10+5)*rand(1); %a
    if x1~=0
       x1=1/x1;%1/cr
       x=[x1;x2;x3];
       g=hw1SUMTcon(x);
       %check if all are negative
       sat = find(g<=0);%satisfied constraint
       if length(sat)==length(g)
           notfound=0;
           display(x);
       end
    end
    i=i+1;
end
i-1
%{
result
x0=[0.475058013560474;...
   13.235315387885350;...
   9.002524127606579];

i=4
%}