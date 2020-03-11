% ECE 580 HW3
% Rahul Deshmukh 
% deshmuk5@purdue.edu
clc; clear all; close all;
format short;
%% include paths
addpath('../OptimModule/line_search/');
addpath('../OptimModule/optimizers/unc/');
addpath('../OptimModule/optimizers/unc/QuasiNewton/');
save_dir = './pix/';

%% Rastrigin Function
fun2plot = @(x1,x2) 20 + (x1/10).^2 + (x2/10).^2 -10*(cos(2*pi*x1/10) + cos(2*pi*x2/10));
grad = @(x1,x2) [ (x1/50) +10*(2*pi/10)*(sin(2*pi*x1/10));...
                   (x2/50) +10*(2*pi/10)*(sin(2*pi*x2/10))];
              
fun = @(x) 20 + (x(1)/10)^2 + (x(2)/10)^2 -10*(cos(2*pi*x(1)/10) + cos(2*pi*x(2)/10));
grad = @(x) [ (x(1)/50)+10*(2*pi/10)*(sin(2*pi*x(1)/10));...
                  (x(2)/50)+10*(2*pi/10)*(sin(2*pi*x(2)/10))];              
% initial guesses
xa = [7.5; 9.0]; xb = [-7.0; -7.5];
              
%% Problem 1: Steepest Descent
fprintf('\n--------------------SD------------------\n');
[x_str1_sd, history1_sd] = steepest_descent(xa, fun, grad);
print_table(history1_sd);x_str1_sd
[x_str2_sd, history2_sd] = steepest_descent(xb, fun, grad);
print_table(history2_sd);x_str2_sd

% plotting Steepest descent paths
a = -15; b = 15;
x = linspace(a,b,20);
y = linspace(a,b,20);
[X, Y] = meshgrid(x,y);
Z = fun2plot(X,Y);

fig1 = figure(1);
hold on;
lvl_list1_sd = plot_traj(history1_sd,'red','o','x^0_a');
lvl_list2_sd = plot_traj(history2_sd, 'black','*','x^0_b');
lvl_list_sd = make_lvl_set(lvl_list1_sd,lvl_list2_sd);
contour(X,Y,Z,lvl_list_sd);
hold off;
xlabel('X'); ylabel('Y');
title('Steepest Descent Trajectory')
xlim([a,b]);
ylim([a,b]);
xticks(a:5:b);
yticks(a:5:b);
box('on');
saveas(fig1,strcat(save_dir,'plot_sd'),'epsc')

%% Problem 2: CG Powell 
fprintf('\n--------------------CG-------------------\n');
[x_str1_cg, history1_cg] = CG(xa, fun, grad);
print_table(history1_cg);x_str1_cg
[x_str2_cg, history2_cg] = CG(xb, fun, grad);
print_table(history2_cg);x_str2_cg

fig2 = figure(2);
hold on;
lvl_list1_cg = plot_traj(history1_cg,'red','o','x^0_a');
lvl_list2_cg = plot_traj(history2_cg, 'black','*','x^0_b');
lvl_list_cg = make_lvl_set(lvl_list1_cg,lvl_list2_cg);
contour(X,Y,Z,lvl_list_cg);
hold off;
xlabel('X'); ylabel('Y');
title('CG Trajectory')
xlim([a,b]);
ylim([a,b]);
xticks(a:5:b);
yticks(a:5:b);
box('on');
saveas(fig2,strcat(save_dir,'plot_cg'),'epsc')

%% Problem 3: Rank one correction (SRS) Algo
fprintf('\n--------------------SRS-------------------\n');
[x_str1_srs, history1_srs] = SRS(xa, fun, grad);
print_table(history1_srs);x_str1_srs
[x_str2_srs, history2_srs] = SRS(xb, fun, grad);
print_table(history2_srs);x_str2_srs

fig3 = figure(3);
hold on;
lvl_list1_srs = plot_traj(history1_srs,'red','o','x^0_a');
lvl_list2_srs = plot_traj(history2_srs, 'black','*','x^0_b');
lvl_list_srs = make_lvl_set(lvl_list1_srs,lvl_list2_srs);
contour(X,Y,Z,lvl_list_srs);
hold off;
xlabel('X'); ylabel('Y');
title('SRS Trajectory')
xlim([a,b]);
ylim([a,b]);
xticks(a:5:b);
yticks(a:5:b);
box('on');
saveas(fig3,strcat(save_dir,'plot_srs'),'epsc')


%% Problem 4: DFP Algo
fprintf('\n--------------------DFP-------------------\n');
[x_str1_dfp, history1_dfp] = DFP(xa, fun, grad);
print_table(history1_dfp);x_str1_dfp
[x_str2_dfp, history2_dfp] = DFP(xb, fun, grad);
print_table(history2_dfp);x_str2_dfp

fig4 = figure(4);
hold on;
lvl_list1_dfp = plot_traj(history1_dfp,'red','o','x^0_a');
lvl_list2_dfp = plot_traj(history2_dfp, 'black','*','x^0_b');
lvl_list_dfp = make_lvl_set(lvl_list1_dfp,lvl_list2_dfp);
contour(X,Y,Z,lvl_list_dfp);
hold off;
xlabel('X'); ylabel('Y');
title('DFP Trajectory')
xlim([a,b]);
ylim([a,b]);
xticks(a:5:b);
yticks(a:5:b);
box('on');
saveas(fig4,strcat(save_dir,'plot_dfp'),'epsc')


%% Problem 5: BFGS Algo
fprintf('\n--------------------BFGS-------------------\n');
[x_str1_bfgs, history1_bfgs] = BFGS(xa, fun, grad);
print_table(history1_bfgs);x_str1_bfgs
[x_str2_bfgs, history2_bfgs] = BFGS(xb, fun, grad);
print_table(history2_bfgs);x_str2_bfgs

fig5 = figure(5);
hold on;
lvl_list1_bfgs = plot_traj(history1_bfgs,'red','o','x^0_a');
lvl_list2_bfgs = plot_traj(history2_bfgs, 'black','*','x^0_b');
lvl_list_bfgs = make_lvl_set(lvl_list1_bfgs,lvl_list2_bfgs);
contour(X,Y,Z,lvl_list_bfgs);
hold off;
xlabel('X'); ylabel('Y');
title('BFGS Trajectory')
xlim([a,b]);
ylim([a,b]);
xticks(a:5:b);
yticks(a:5:b);
box('on');
saveas(fig5,strcat(save_dir,'plot_bfgs'),'epsc')


%% Local plotting function
function lvl_list = plot_traj(history,color,marker, disp_text)
    lvl_list = [];
    for i=1:history.Niters -1
        x = history.data(i).x(1);
        y = history.data(i).x(2);
        u = history.data(i+1).x(1);
        v = history.data(i+1).x(2);
        plot([x,u],[y,v],'color',color,'marker',marker); 
        lvl_list = [lvl_list, history.data(i).f_k];
    end
    lvl_list = [lvl_list, history.data(i+1).f_k];
    text(history.data(1).x(1),history.data(1).x(2),disp_text, 'color',color, 'FontSize',15);    
end

function L = make_lvl_set(list1, list2)    
    Lmin = min(min(list1), min(list2));
    Lmax = max(max(list1), max(list2));
    temp = linspace(Lmin,Lmax,10);
    L = [list1, list2, temp];
end
% 
% function print_table(history)
%     rowvec_fmt = ['[', repmat('%0.4f, ', 1, numel(history.data(1).x)-1), '%0.4f]'];
%     fprintf('Iter(k)\t x_k\t f_k\t g_k\t alpha_k\n');
% 
%     for k=1:history.Niters
%         fprintf('%d\t',k);
%         fprintf(rowvec_fmt,history.data(k).x); fprintf('\t');
%         fprintf('%0.4f\t',history.data(k).f_k);
%         fprintf(rowvec_fmt,history.data(k).g_k); fprintf('\t');        
%         fprintf('%0.4f\t',history.data(k).alpha_k);fprintf('\n');
%     end        
% end

function print_table(history)
    rowvec_fmt = ['[', repmat('%0.4f, ', 1, numel(history.data(1).x)-1), '%0.4f]'];
    fprintf('\\hline\n');
    fprintf('Iter(k)&\t x_k&\t f_k&\t g_k&\t alpha_k\\\\\n');
    fprintf('\\hline\n');
    for k=1:history.Niters
        fprintf('%d&\t',k);
        fprintf(rowvec_fmt,history.data(k).x); fprintf('&\t');
        fprintf('%0.4f&\t',history.data(k).f_k);
        fprintf(rowvec_fmt,history.data(k).g_k); fprintf('&\t');        
        fprintf('%0.4f\t',history.data(k).alpha_k);fprintf('\\\\\n');
        fprintf('\\hline\n');
    end        
end