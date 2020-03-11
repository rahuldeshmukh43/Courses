function phi=pseudo_obj_ip(f,g,x,rp)
% input: f: is a funciton handle, 
%       g: cell array of constraint function handles
%       x: vector
%       rp:penalty multiplier
% output: phi: scalar value psuedo objectivbe function for classic
% intereior penalty method

temp=0;
for j=1:length(g)
   gj=g{j};
   temp=temp+(-1/gj(x));
end
phi=f(x)+rp*(temp);
end