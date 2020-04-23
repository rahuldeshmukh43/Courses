function offspring = single_parent_crossover(mating_pool, p_xover, Num_var, coded_lens)
[N_pop,~] = size(mating_pool);
cumsum_coded_lens = [0, cumsum(coded_lens)];

rand_nums = rand(1,N_pop);
do_xover = rand_nums > (1-p_xover);
offspring = mating_pool;
for k = 1:N_pop
    if do_xover(k)
        k_offspring = offspring(k,:);
        js = randi([1,Num_var],1,2);
        j1_idx = cumsum_coded_lens(js(1)) + 1 : cumsum_coded_lens(js(1)+1);
        j2_idx = cumsum_coded_lens(js(2)) + 1 : cumsum_coded_lens(js(2)+1);
        j1_code = k_offspring(j1_idx);
        k_offspring(j1_idx) = k_offspring(j2_idx);
        k_offspring(j2_idx) = j1_code;
        offspring(k,:) = k_offspring;
    end
end
end