function mutated = mutation(parents, p_mut)
[N_pop,~] = size(parents);
rand_nums = rand(N_pop,1);
do_mut_idx = find(rand_nums < p_mut);
mutated = parents;
% complement each bit in parent
mutated(do_mut_idx,:) = 1-parents(do_mut_idx,:);
end