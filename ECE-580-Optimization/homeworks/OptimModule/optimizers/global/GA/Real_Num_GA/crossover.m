function offspring = crossover(mating_pool,p_xover, method)
switch nargin
    case 2
        method='av';
end
%shuffle mating pool
[Npop,~] =  size(mating_pool);
mating_pool = mating_pool(randperm(Npop), :);
rand_num = rand(round(Npop/2),1);
do_xover = rand_num < p_xover;
do_xover_idx = find(do_xover);

alpha = rand(length(do_xover_idx),1);
offspring = mating_pool;
if strcmp(method,'av')
    for i = 1:length(do_xover_idx)
        k = do_xover_idx(i);
        parents = mating_pool([2*k-1 2*k],:);
        offspring([2*k-1 2*k], :) = [sum(parents, 1)/2.0;
                                    parents((alpha(i)> 0.5)+1,:)];
    end
elseif strcmp(method,'conv_combo')
    for i = 1:length(do_xover_idx)
        k = do_xover_idx(i);
        parents = mating_pool([2*k-1 2*k],:);
        offspring([2*k-1 2*k], :) = [alpha(i), 1-alpha(i);
                                     1-alpha(i), alpha(i)]*parents;
    end
else
    error('Method for crossover not implemented');
end

end