% this file provides input for and calls fminsearch to use the Nelder-Mead
% Simplex method
% Modified on 11/05/07 by Bill Crossley.
clc;
close all;
clear all;

options = optimset('Display','iter');

x0 = [3,-2];

[x,fval,exitflag,output] = fminsearch('NMfunc',x0,options)

fprintf('--------------------------------------------------------');
% restart: second call to fminsearch
x0=x;
[x,fval,exitflag,output] = fminsearch('NMfunc',x0,options)

%  plot of function and optimum point
t=-2:1/10:2;
[X,Y]=meshgrid(t,t);
f =@(x) [1+(x(1)+x(2)+1)^2*(19-14*x(1)+3*x(1)^2-14*x(2)+6*x(1)*x(2)+3*x(2)^2)]...
    *[30+(2*x(1)-3*x(2))^2*(18-32*x(1)+12*x(1)^2+48*x(2)-36*x(1)*x(2)+27*x(2)^2)];

Z=zeros(size(t,2),size(t,2));
for i=1:size(Z,1)
    for j = 1:size(Z,2)
        Z(i,j)=f([X(i,j),Y(i,j)]);
    end
end

figure(1);
surf(X,Y,Z)
xlabel('x1');
ylabel('x2');
zlabel('function');

msize=100;
figure(2);
contour(X,Y,Z,50);hold on;
colorbar;
scatter(x0(1),x0(2),msize,'o','g');
% text(x0(1),x0(2),'X*1')
scatter(x(1),x(2),msize,'^','b');
% text(x(1),x(2),'X*2')
x_star=[0,-1];
scatter(x_star(1),x_star(2),msize,'*','r');
% text(x_star(1),x_star(2),'X*')
xlabel('x1');
ylabel('x2');
