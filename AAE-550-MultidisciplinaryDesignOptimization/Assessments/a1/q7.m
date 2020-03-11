clc; clear all; close all;
format short
%golden search
f=@(x) 2*x^3+27*x^2+3;
% a<= x <=b
a=-4;b=1;
%scale doamin to 0 to 1
sf=@(x) 2*((b-a)*x+a)^3+27*((b-a)*x+a)^2+3;
t=(3-sqrt(5))/2;
%part A
[x1,x2]=findpts(a,b);
%part B
tempf = [f(a),f(x1),f(x2),f(b)];
[minf,imin]=min(tempf);

N=100;
tol=1e-6;
notconverged=true;
count=1;
while count<=N && notconverged
    if f(x1)<f(x2)
        b=x2;
        [x1,x2]=findpts(a,b);
        display(strcat(num2str(count),' new point ',num2str(x1),'with function value ',num2str(f(x1))));
        display([a,x1,x2,b]);
    else
        a=x1;
        [x1,x2]=findpts(a,b);   
        display(strcat(num2str(count),' new point ',num2str(x2),'with function value ',num2str(f(x2))));
        display([a,x1,x2,b]);
    end
    
    if abs(b-a)<tol
        notconverged=false;
    end
%     display(strcat('iteration ',num2str(count),' x1 is ',num2str(x1),...
%         ' and x2 is ',num2str(x2)));
    count=count+1;    
end
