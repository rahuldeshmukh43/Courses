% phi(alpha) for SQP step length problem
function [output]=phi(a,x,s,u)
% Input: a= alpha, the step length
%        f=objective function value at x+as : scalar
%        g=inequality constraints at x+as; col vector size m
%        h=equality constraints at x+asl col vector size l 
%        u= penalty multipliers [ineq,eq] only the active ones
xq=x+a*s;
f=fun(xq);
[g,h]=con(xq);
m=length(g);
l=length(h);
ineq_term=sum(u(1:m).*(max(0,g)));
eq_term=sum(u(m:m+l).*(abs(h)));

output=f+ineq_term+eq_term;

end