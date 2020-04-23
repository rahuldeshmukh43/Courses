function [g,geq]=NonLcon(x)

g=zeros(2,1);

g(1)=-2.5+2.5*(x(1)-2)^3+x(2);
g(2)=-3.85-4*(x(2)-x(1)+0.5)^2+x(1)+x(2);

geq=[];

end