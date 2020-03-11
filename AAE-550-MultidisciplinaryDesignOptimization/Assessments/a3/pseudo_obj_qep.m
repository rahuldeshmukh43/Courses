function phi = pseudo_obj_qep(f,g,x,rp)
% input: f: is a funciton handle, 
%       g: cell array of constraint function handles
%       x: vector
%       rp:penalty multiplier
% output: phi: scalar value psuedo objectivbe function 

temp=0;
for i=1:length(g)
   tempfun=g{i};
   temp=temp+(max(0,tempfun(x)))^2;
end
phi=f(x)+rp*(temp);

end