function [best_f, av_f, worse_f] = log_f(f_parent, best_f, av_f, worse_f)
best_f = [best_f, max(f_parent)];
av_f = [av_f, mean(f_parent)];
worse_f = [worse_f, min(f_parent)];
end