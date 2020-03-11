%1.2 
% plot func_one over x = (0,3*pi)
x = 0:0.1:3*pi;
y = func_one(x);
figure(1);
plot(x,y,'-ro'); title('1.2 plot of function one');
xlabel('x');ylabel('y');