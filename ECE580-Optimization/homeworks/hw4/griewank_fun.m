function y = griewank_fun(X_swarm,min_bool)
%     d dimensionalfriewank function 
switch nargin
    case 1
        min_bool=1;
end

[x_dim ,Nswarm] = size(X_swarm);
y = zeros(Nswarm,1);
for k=1:Nswarm
    sum = 0;
    prod = 1;
    x = X_swarm(:,k);
    for i=1:x_dim
       x_i = x(i);
       sum = sum + x_i^2/4000;
       prod = prod * cos(x_i/sqrt(i));    
    end
    y(k) = sum - prod +1;
end
if ~min_bool
    y = -1*y;
end
end