function X_coded = encode(X, lb, ub, code_lens)
[N_pop,~] = size(X);
L = sum(code_lens);
cumsum_code_lens = [0, cumsum(code_lens)];
X_coded = zeros(N_pop,L);
Num_var = length(lb);
for i = 1:N_pop
    x = X(i,:) - lb;
    x_coded = zeros(1,L);
    for j = 1:Num_var
       xj = x(j);
       x_coded(cumsum_code_lens(j) + 1 : cumsum_code_lens(j+1)) = de2bi(xj ,code_lens(j));
    end
    X_coded(i,:) = x_coded;
end