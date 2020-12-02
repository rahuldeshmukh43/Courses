function mutated = mutation(parents, p_mut, lb, ub)
[N_pop,~] = size(parents);
rand_nums = rand(N_pop,1);
do_mut_idx = find(rand_nums < p_mut);
mutated = parents;
alpha = rand(length(do_mut_idx),1);
w = rand(length(do_mut_idx),length(lb));
% scale and translate w to domain
w = w.*(ub-lb) + lb;
mutated(do_mut_idx,:) = parents(do_mut_idx,:).*alpha + w.*(1-alpha);
end