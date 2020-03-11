%plot func_two over [-2 6]x[-2 6]
a = -2;b = 6; N =10;
x = linspace(a,b,N);
[X,Y]=meshgrid(x,x);
figure(1);
Z =zeros(N);
for i=1:N
    for j = 1:N
        Z(i,j)= func_two([X(i,j);Y(i,j)]);
    end
end
figure(1);
surf(X,Y,Z);title('Two variable function');
xlabel('x1');ylabel('x2');zlabel('f(x)');

figure(2);% for contour plots
[c,h]=contour(X,Y,Z);
clabel(c,h); xlabel('x1');ylabel('x2');
title('Contours of function two');