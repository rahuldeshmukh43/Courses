function alm=pseudo_obj_alm(x,f,g,h,rp,lam)
%input: x: vector 
%       f: function handle for objective function
%       g: ineauality constraints, cell array of function handles
%       h: eauality constraints, cell array of function handles
%       rp: Penalty multiplier
%       lam: lagrangian multipliers vector ordered [lam_ine lam_eq]
% output: alm: augmented lagrangian pseudo objective function value

sum_ineq=0;
for j=1:length(g)
    gj=g{i};
    psi_j=max(gj(x),-lam(j)/(2*rp));
   sum_ineq= sum_ineq+(lam(j)*psi_j+rp*psi_j^2);
end

m=lenght(g);
sum_eq=0;
for k=1:length(h)
   hk=h{k};
   sum_eq=sum_eq+(lam(k+m)*hk(x)+rp*(hk(x))^2);
end

alm=f(x)+sum_ineq+sum_eq;

end