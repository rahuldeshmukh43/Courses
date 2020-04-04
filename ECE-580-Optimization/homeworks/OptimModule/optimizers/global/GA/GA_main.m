% ECE 580 HW4: Problem 4
% Rahul Deshmukh 
% deshmuk5@purdue.edu
clc; clear all; close all;
format long;
save_dir = '../../../../hw4/';
%% TSP setup
% map coordinates
x_pos = [ 0.4306
 3.7094
 6.9330
 9.3582
 4.7758
 1.2910
 4.83831
 9.4560
 3.6774
 3.2849];

y_pos = [ 7.7288
 2.9727
 1.7785
 6.9080
 2.6394
 4.5774
 8.43692
 8.8150
 7.0002
 7.5569];

Num_city = length(x_pos);
lb = 1*ones(1,Num_city);
ub = Num_city*ones(1,Num_city);
resolution = ones(1,Num_city);
coded_lens = ceil(log2((ub-lb)./resolution));


%% GA: solver params
total_possible_path = factorial(Num_city)
N_pop = 1000;
p_xover = 0.8;
p_mut = 0.05;
Niters = 200;
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

% draw initial population: all possible permuations of route
s = RandStream('mlfg6331_64'); 
X = zeros(N_pop, Num_city);
for i=1:N_pop
   ith_route =  datasample(s, 1:Num_city, Num_city, 'Replace', false);
   X(i,:) = ith_route;    
end
%encode X
parents = encode(X, lb, ub, coded_lens);
% evaluate fitness of parents
f_parent = -1*fitness(parents, lb, coded_lens, resolution, x_pos, y_pos);
[best_f, av_f, worse_f] = log_f(f_parent, best_f, av_f, worse_f);
for i=1:Niters
   % generate mating pool using selection
   mating_pool = selection(parents, f_parent);
   %perform crossover
   parents = crossover(mating_pool, p_xover, Num_city, coded_lens);
   %perform mutation
   
   %perform elitism
   parents = elitism(parents, f_parent);
   %evaluate fitness of offspring
   f_parent = -1*fitness(parents, lb, coded_lens, resolution, x_pos, y_pos);  
   [best_f, av_f, worse_f] = log_f(f_parent, best_f, av_f, worse_f);
end
% find the best offspring
[f_star, k_star] = max(f_parent);
fprintf(strcat('Shortest Route Lenght: ',num2str(-1*f_star)))
x_star_coded = parents(k_star,:);
x_star = decode(x_star_coded, lb, coded_lens, resolution)

%% Convergence Plotting
fig1 = figure(1);
hold on; grid on;
x = 1:Niters+1;
h1 =plot(x,-1*av_f,'-b','LineWidth',2);
h2 = plot(x,-1*best_f,'-r','LineWidth',2);
h3 = plot(x,-1*worse_f,'-k', 'LineWidth',2);
v = 1:10:Niters+1;
plot(x(v),-1*av_f(v),'bx');
plot(x(v),-1*best_f(v),'ro');
plot(x(v),-1*worse_f(v),'k*');
legend([h1,h2,h3],{'Average','Best','Worse'},'Location','northeast');
hold off;
box('on');
xlabel('Num Iters'); ylabel('Function value');
title('Convergence of GA');
saveas(fig1,strcat(save_dir,'ga_conv'),'epsc');
%% Route plotting
fig2 = figure(2);
hold on;grid on;
scatter(x_pos,y_pos,'ob');
x_star_end = [x_star(2:end), x_star(1)];
for i=1:Num_city
    x = x_pos(x_star(i));
    y = y_pos(x_star(i));
    u = x_pos(x_star_end(i)) - x;
    v = y_pos(x_star_end(i)) - y;
    text(x,y,num2str(x_star(i)),'FontSize',18, 'FontWeight','bold',...
        'HorizontalAlignment','left', 'VerticalAlignment','middle' );
    quiver(x,y,u,v,'r','Autoscale','off','LineWidth',2);
end
box('on');hold off;
xlabel('X'); ylabel('Y');
saveas(fig2,strcat(save_dir,'ga_best_route'),'epsc');
title('Optimal Route')

