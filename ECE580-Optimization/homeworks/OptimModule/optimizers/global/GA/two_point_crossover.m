function offspring = two_point_crossover(mating_pool, p_xover)

[N_pop,L] = size(mating_pool);
% shuffle parents
mating_pool = mating_pool(randperm(N_pop),:);
% generate rand nums for deciding if do crossover?
rand_nums = rand(1,round(N_pop/2));
do_xover = rand_nums > (1-p_xover);
offspring = zeros(N_pop, L);

for k = 1:round(N_pop/2)
    parents = mating_pool([2*k-1 ,2*k],:);
    if do_xover(k)
        % find crossover point
        xover_pt = randi(L,1);
        % switch genes
        offspring(2*k-1,:)= [parents(1,1:xover_pt), parents(2,xover_pt+1:end)];
        offspring(2*k,:) = [parents(2,1:xover_pt), parents(1,xover_pt+1:end)];
    else
        offspring([2*k-1, 2*k],:) = parents;
    end
end

end