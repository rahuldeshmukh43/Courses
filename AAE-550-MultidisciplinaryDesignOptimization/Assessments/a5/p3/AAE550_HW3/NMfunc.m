function phi = NMfunc(x)
% objective function: egg-crate function from class
% linear exterior penalty to enforce bounds
f = x(1)^2 + x(2)^2 + 25 * (sin(x(1))^2 + sin(x(2))^2);
g(1) = x(1) / (-2 * pi) - 1;  % enforces lower bound
g(2) = x(2) / (-2 * pi) - 1;
g(3) = x(1) / (2 * pi) - 1; % enforces upper bound
g(4) = x(2) / (2 * pi) - 1;

P = 0.0;    % initialize penalty function
for i = 1:4
    P = P + 10 * max(0,g(i));  % use c_j = 10 for all bounds
end
phi = f + P;