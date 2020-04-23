function mating_pool = roulette(parent, f_parent)
[N_pop,~] = size(parent);
f_min = min(f_parent);
f = f_parent - f_min;
F = sum(f);
p = f/F;
q = cumsum(p);
rand_nums = rand(N_pop,1);
mating_idx = zeros(N_pop,1);
temp = q' -rand_nums;
for k=1:N_pop
   mating_idx(k) = find(temp(k,:) > 0, 1);
end
mating_pool = parent(mating_idx, :);
end