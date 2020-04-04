% ECE 580 HW4
% Rahul Deshmukh 
% deshmuk5@purdue.edu
clc; clear all; close all;
format short;
%% include paths
addpath('../OptimModule/optimizers/global/');
save_dir = './pix/';

%% Problem 2: PSO min
%plot griewank fun
x = linspace(-5,5,100);
[X,Y] = meshgrid(x,x);
[h,w] = size(X);
Z = zeros(h,w);
for ih=1:h
    for iw=1:w
        Z(ih,iw) = griewank_fun([X(ih,iw);Y(ih,iw)]);
    end
end
fig = figure(1);
surfc(X,Y,Z);grid on;
view(3);
xlabel('X');
ylabel('Y');
saveas(fig,strcat(save_dir,'surf_plot'),'epsc');

a = [-5;-5];
b = [5;5];
[x_star_min, history_min] = particleswarm(@(x)griewank_fun(x), a, b);
x_star_min
fval = history_min.data(history_min.Niters+1).gbest_fval
fig2= figure(2);
hold on;grid on;
pso_conv_plot(history_min,1);
hold off;
box('on');
xlabel('Num Iters'); ylabel('Function value');
title('Convergence of PSO for minimization');
saveas(fig2,strcat(save_dir,'plot_pso_min'),'epsc');

fig3= figure(3);
hold on;grid on;
contour(X,Y,Z);
plot_pso_traj(history_min);
xlabel('X'); ylabel('Y');
title('PSO min solution')
hold off;
xlim([a(1),b(1)]);
ylim([a(2),b(2)]);
xticks(a(1):1:b(1));
yticks(a(2):1:b(2));
box('on');
saveas(fig3,strcat(save_dir,'pso_min_traj'),'epsc')

%% Problem 3: PSO max
[x_star_max, history_max] = particleswarm(@(x)griewank_fun(x,0), a, b);
x_star_max
fval = -1*history_max.data(history_max.Niters+1).gbest_fval
fig4= figure(4);
hold on;grid on;
pso_conv_plot(history_max,0);
hold off;
box('on');
xlabel('Num Iters'); ylabel('Function value');
title('Convergence of PSO for maximization');
saveas(fig4,strcat(save_dir,'plot_pso_max'),'epsc');

fig5= figure(5);
hold on;grid on;
contour(X,Y,Z);
plot_pso_traj(history_max);
xlabel('X'); ylabel('Y');
title('PSO max solution')
hold off;
xlim([a(1),b(1)]);
ylim([a(2),b(2)]);
xticks(a(1):1:b(1));
yticks(a(2):1:b(2));
box('on');
saveas(fig5,strcat(save_dir,'pso_max_traj'),'epsc')

%% Problem 5: Linprog
fprintf('-----------Linear Programming----------------');
A = [1, 2, 1, 2;
    6, 5, 3, 2;
    3, 4, 9, 12];
b = [20; 100; 75];
Aeq= []; beq = [];
lb = [0, 0, 0, 0];
ub = Inf*[1, 1, 1, 1];
c = [6, 4, 7, 5];
[x_star_linprog,fval] = linprog(-1*c, A, b, Aeq, beq, lb, ub);
x_star_linprog
fval

%% Local helper functions for plotting
% plotting for PSO
function pso_conv_plot(history, min_bool)
    av = [];
    gbest = [];
    worse =[];
    for i=1:history.Niters + 1
       av = [av; history.data(i).pbest_av]; 
       gbest = [gbest; history.data(i).gbest_fval];
       worse = [worse; history.data(i).pbest_worse];
    end
    if ~min_bool
       av=-1*av; gbest = -1*gbest; worse = -1*worse; 
    end
    x = 0:1:history.Niters;
    h1 =plot(x,av,'-b','LineWidth',2);
    h2 = plot(x,gbest,'-r','LineWidth',2);
    h3 = plot(x,worse,'-k', 'LineWidth',2);
    v = 1:10:history.Niters+1;
    plot(x(v),av(v),'bx');
    plot(x(v),gbest(v),'ro');
    plot(x(v),worse(v),'k*');
    if min_bool
        legend([h1,h2,h3],{'Average','Best','Worse'},'Location','northeast');
    else
        legend([h1,h2,h3],{'Average','Best','Worse'},'Location','southeast');
    end
end
function plot_pso_traj(history)
best_x = history.data(history.Niters + 1).gbest_x;
plot(best_x(1,:),best_x(2,:),'rx','MarkerSize',20);
end
