function [g,geq]=Goal_Att_cons(x,a,f1,f2,f_g)


gamma=x(end);
x=x(1:end-1);


g=zeros(4,1);
% converted constraints
g(1)= (f1(x)-a(1)*gamma)/f_g(1)-1;
g(2)= (f2(x)-a(2)*gamma);

% hard constraints
g(3)=-2.5+2.5*(x(1)-2)^3+x(2);
g(4)=-3.85-4*(x(2)-x(1)+0.5)^2+x(1)+x(2);

geq=[];
end