%plot func_two over [-2 6]x[-2 6]
clc; clear all; close all;
a = -2;b = 6; N =10;
x = linspace(a,b,N);
[X,Y]=meshgrid(x,x);
figure(1);
Z =zeros(N);
g1=zeros(N);
g2 = zeros(N);
for i=1:N
    for j = 1:N
        Z(i,j)= func_two([X(i,j);Y(i,j)]);
        temp=cons_three([X(i,j);Y(i,j)]);
        g1(i,j)=temp(1);
        g2(i,j)=temp(2);
    end
end
figure(1);
surf(X,Y,Z);title('Two variable function');
xlabel('x1');ylabel('x2');zlabel('f(x)');

figure(2);% for contour plots
hold on;
% [c,h]=contour(X,Y,Z);
[c,h]=contour(X,Y,Z,[0,0.25,0.5,1]);
clabel(c,h); xlabel('x1');ylabel('x2');
title('Contours of function two');
[cg1,hg1]=contour(X,Y,g1,[0 0]);clabel(cg1,hg1);
[cg2,hg2]=contour(X,Y,g2,[0 0]);clabel(cg2,hg2);

% x* = [1,5/3]; visually found

%optimtool results

% exitflag=1 reported; means local minima found
% iterations = 5
% gradient = [1.20713496199558;3.62140499432140];