function new_pop = elitism(pop, fitness)
new_pop = pop;
temp_fit = fitness;
[~, max_fit_idx] = max(temp_fit);
temp_fit(max_fit_idx) = min(temp_fit);
[~, other_max_fit_idx] = max(temp_fit);
new_pop([1,2],:) = pop([max_fit_idx, other_max_fit_idx],:);
end