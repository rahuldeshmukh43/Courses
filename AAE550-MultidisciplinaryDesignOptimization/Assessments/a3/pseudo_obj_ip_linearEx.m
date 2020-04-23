function phi=pseudo_obj_ip_linearEx(f,g,x,rp,ep)
% input: f: is a funciton handle, 
%       g: cell array of constraint function handles
%       x: vector
%       rp:penalty multiplier
% output: phi: scalar value psuedo objectivbe function for classic
% intereior penalty method with linear extension

temp=0;
for j=1:length(g)
   tempfun=g{j};
   gj=tempfun(x);
   if gj<ep
       gj_hat= - (1/gj);
   else
       gj_hat=-((2*ep-gj)/(ep^2));
   end
   temp=temp+(gj_hat);
end
phi=f(x)+rp*(temp);
end