function [g,geq]=NonLcon(x,a,fl_min,f1,f2)

beta=x(end);
x=x(1:end-1);

iphi=zeros(2,1);
iphi(1)= ((f1(x)-fl_min(1))/fl_min(1))^2;
iphi(2)= ((f2(x)-fl_min(2))/fl_min(2))^2;

g = a.*iphi -beta*[1;1];

geq=[];
end