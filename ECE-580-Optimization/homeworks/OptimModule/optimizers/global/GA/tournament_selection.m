function mating_pool = tournament_selection(parent, f_parent, method)
[N_pop,~] = size(parent);
mating_idx = zeros(N_pop,1);
if method == 1
   a = randi([1, N_pop], 1, N_pop) ;
   b = randi([1, N_pop], 1, N_pop) ;
   fa = f_parent(a);
   fb = f_parent(b);
   for k=1:N_pop
       if fa(k)>fb(k)
           mating_idx(k) = a(k);
       else
           mating_idx(k) = b(k);
       end
   end
elseif method == 2
    a = randi([1, N_pop], 1, N_pop);
    fa = f_parent(a);
    for k=1:N_pop
       if fa(k)>f_parent(k)
           mating_idx(k) = a(k);
       else
           mating_idx(k) = k;
       end
   end
end    
mating_pool = parent(mating_idx, :);
end