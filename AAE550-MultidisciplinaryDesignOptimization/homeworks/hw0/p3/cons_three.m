function g = cons_three(x)
%input x is a col vector of size=2
%evaluates the two convex constraints g1 and g2
%o/p g is a vector with [g1(x) g2(x)]
g =zeros(length(x),1);
g(1)=6-x(1)-3*x(2);
g(2)= 1-x(1);

