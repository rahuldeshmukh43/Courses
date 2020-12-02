% ECE 580 HW5: Problem 1
% Rahul Deshmukh 
% deshmuk5@purdue.edu
clc; clear all; close all;
format short;
%% Include paths
addpath('../OptimModule/optimizers/global/GA/');
save_dir = './pix/';
%% set seed
rng(7); % tried 0:10, 7 gave the smallest result, can jump between global and local min
%% Canonical GA problem setup
lb = -5*ones(1,2);
ub = 5*ones(1,2);
Num_vars = length(lb);
bits = 10;
coded_lens = bits*ones(1,Num_vars);
resolution = (ub-lb)./(2.^coded_lens-1)

%% GA: solver params
N_pop = 40; % cant be odd integer
p_xover = 0.9;
p_mut = 0.01;
Niters = 30;
selection_method = 'tournament_method2';

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
% discretize to resolution
X = floor((X - lb)./resolution).*resolution + lb;

%encode X
parents = encode(X, lb, ub, coded_lens, resolution);
% evaluate fitness of parents
f_parent = -1*fitness_griewank(parents, lb, coded_lens, resolution);
[best_f, av_f, worse_f] = log_f(f_parent, best_f, av_f, worse_f);
for i=1:Niters
   % generate mating pool using selection
   mating_pool = selection(parents, f_parent);
   %perform crossover
   parents = two_point_crossover(mating_pool, p_xover);
   %perform mutation
   parents = mutation(parents, p_mut);
   %perform elitism
   parents = elitism(parents, f_parent);
   %evaluate fitness of offspring
   f_parent = -1*fitness_griewank(parents, lb, coded_lens, resolution);  
   [best_f, av_f, worse_f] = log_f(f_parent, best_f, av_f, worse_f);
end
% find the best offspring
[f_star, k_star] = max(f_parent);
fprintf(strcat('best fval: \t',num2str(-1*f_star)))
x_star_coded = parents(k_star,:);
x_star = decode(x_star_coded, lb, coded_lens, resolution)

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
title('Convergence of GA');
saveas(fig1,strcat(save_dir,'ga_canon_conv'),'epsc');