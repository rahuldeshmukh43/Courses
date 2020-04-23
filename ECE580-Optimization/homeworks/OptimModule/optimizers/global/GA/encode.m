function X_coded = encode(X, lb, ub, code_lens, resolution)
[N_pop,~] = size(X);
L = sum(code_lens);
cumsum_code_lens = [0, cumsum(code_lens)];
X_coded = zeros(N_pop,L);
Num_var = length(lb);
% convert discretized X to integers for encoding
X = round((X - lb)./resolution);
for i = 1:N_pop
    x = X(i,:);
    x_coded = zeros(1,L);
    for j = 1:Num_var
       xj = x(j);
       x_coded(cumsum_code_lens(j) + 1 : cumsum_code_lens(j+1)) = de2bi(xj ,code_lens(j));
    end
    X_coded(i,:) = x_coded;
end