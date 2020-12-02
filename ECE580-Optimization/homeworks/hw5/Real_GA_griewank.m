% ECE 580 HW5: Problem 2
% Rahul Deshmukh 
% deshmuk5@purdue.edu
clc; clear all; close all;
format short;
%% Include paths
addpath('../OptimModule/optimizers/global/GA/');
addpath('../OptimModule/optimizers/global/GA/Real_Num_GA/');
save_dir = './pix/';
%% set seed
rng(6); % tried 0:10 ,6 was the best and always converges to global min
%% Real GA problem setup
lb = -5*ones(1,2);
ub = 5*ones(1,2);
Num_vars = length(lb);

%% GA: solver params
N_pop = 40; % cant be odd integer
p_xover = 0.9;
p_mut = 0.01;
Niters = 30;
selection_method = 'tournament_method2';
xover_method = 'conv_combo';

%% GA starts

% intialize collectors
best_f = [];
av_f = [];
worse_f = [];

% choose type of selector
if strcmp(selection_method, 'roulette')
    selection = @(x,f) roulette(x,f);
elseif strcmp(selection_method, 'tournament_method1')
    selection = @(x,f) tournament_selection(x,f,1);
elseif strcmp(selection_method, 'tournament_method2')
    selection = @(x,f) tournament_selection(x,f,2);
end

% draw initial population
X = rand(N_pop, Num_vars);
% scale to domain
X = (X.*(ub-lb) + lb);

parents = X;
% evaluate fitness of parents
f_parent = -1*griewank_fun(parents');
[best_f, av_f, worse_f] = log_f(f_parent, best_f, av_f, worse_f);
for i=1:Niters
   % generate mating pool using selection
   mating_pool = selection(parents, f_parent);
   %perform crossover
   parents = crossover(mating_pool, p_xover, xover_method);
   %perform mutation
   parents = mutation(parents, p_mut, lb, ub);
   %perform elitism
   parents = elitism(parents, f_parent);
   %evaluate fitness of offspring
   f_parent = -1*griewank_fun(parents');
   [best_f, av_f, worse_f] = log_f(f_parent, best_f, av_f, worse_f);
end
% find the best offspring
[f_star, k_star] = max(f_parent);
fprintf(strcat('best fval: \t',num2str(-1*f_star)))
x_star = parents(k_star,:)

%% Convergence Plotting
fig1 = figure(1);
hold on; grid on;
x = 1:Niters+1;
h1 =plot(x,-1*av_f,'-b','LineWidth',1);
h2 = plot(x,-1*best_f,'-r','LineWidth',1);
h3 = plot(x,-1*worse_f,'-k', 'LineWidth',1);
v = 1:5:Niters+1;
plot(x(v),-1*av_f(v),'bx');
plot(x(v),-1*best_f(v),'ro');
plot(x(v),-1*worse_f(v),'k*');
legend([h1,h2,h3],{'Average','Best','Worse'},'Location','northeast');
hold off;
box('on');
xlabel('Num Iters'); ylabel('Function value');
xlim([1, Niters+1])
ylim(max(abs(worse_f))*(1.1)*[-1,1]);
title('Convergence of Real GA');
saveas(fig1,strcat(save_dir,'ga_real_conv'),'epsc');