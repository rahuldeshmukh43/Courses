function [x1,x2]=findpoints(a,b)
%returns x1 and x2
t=(3-sqrt(5))/2;
x1=(b-a)*t+a;
x2=(b-a)*(1-t)+a;
end