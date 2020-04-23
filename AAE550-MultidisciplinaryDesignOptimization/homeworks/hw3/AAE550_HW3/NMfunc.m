function phi = NMfunc(x)
% objective function: Goldstein Price Function
% linear exterior penalty to enforce bounds

f = [1+(x(1)+x(2)+1)^2*(19-14*x(1)+3*x(1)^2-14*x(2)+6*x(1)*x(2)+3*x(2)^2)]...
    *[30+(2*x(1)-3*x(2))^2*(18-32*x(1)+12*x(1)^2+48*x(2)-36*x(1)*x(2)+27*x(2)^2)];



g(1) = -x(1) / (2) - 1;  % enforces lower bound
g(2) = -x(2) / (2) - 1;
g(3) = x(1) / (2) - 1; % enforces upper bound
g(4) = x(2) / (2) - 1;

P = 0.0;    % initialize penalty function
for i = 1:4
    P = P + 1 * max(0,g(i));  % use c_j = 10 for all bounds
end

rp=1E6;
phi = f + rp*P;

end