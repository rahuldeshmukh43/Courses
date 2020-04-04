function [best_f, av_f, worse_f] = log_f(f_parent, best_f, av_f, worse_f)

[best, best_id] = max(f_parent);
av = mean(f_parent);
worse = min(f_parent);

best_f = [best_f, best];
av_f = [av_f, av];
worse_f = [worse_f, worse];
end