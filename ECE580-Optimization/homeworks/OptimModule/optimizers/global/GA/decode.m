function X = decode(X_coded, lb, code_lens, resolution)
[N_pop,~] = size(X_coded);
L = sum(code_lens);
cumsum_code_lens = [0, cumsum(code_lens)];
Num_var = length(lb);
X = zeros(N_pop,Num_var);
for i=1:N_pop
   x_coded = X_coded(i,:);
   x = zeros(1,Num_var);
   for j=1:Num_var
       xj_coded = x_coded(cumsum_code_lens(j) + 1 : cumsum_code_lens(j+1));
       x(j) = resolution(j)*bi2de(xj_coded);
   end
   X(i,:) = x + lb;
end
end