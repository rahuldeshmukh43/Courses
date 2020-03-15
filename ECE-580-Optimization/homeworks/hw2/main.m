% ECE 580 HW2 
% Rahul Deshmukh 
% deshmuk5@purdue.edu
clc; clear all; close all;
format short;
%% include paths
addpath('./line_search/');
addpath('./optimizers/');
save_dir = './pix/';

%% Problem 3: Identifying uncertainity interval
fun1 = @(x) (0.5)*x'*[2,1;1,2]*x;
grad1 = @(x) [2,1;1,2]*x;
step_size0 = 0.1;

x0 = [0.8, -0.25]';

fprintf('The uncertainity interval for problem 3 is [xa, xb]\n');
[xa, xb, P3_history] = get_search_interval(x0, fun1, -1*grad1(x0), step_size0)

%% Problem 4: P3 with Golden section
TolX = 0.05;
[xa_gs, xb_gs, P4_history] = golden_section(xa, xb, fun1, TolX, 1)

%% Problem 5: P3 with fibonacci method
[xa_fs, xb_fs, P5_history] = fibonacci_method(xa, xb, fun1, TolX, 1)

%% Problem 6: P3 with newton method
xa = x0;

Hessian1 = @(x) [2,1;1,2];
phi = @(a) fun1(xa - a*grad1(xa));
dphi = @(a) 2*a*fun1(grad1(xa)) - grad1(xa)'*Hessian1(xa)*xa;
ddphi = @(a) 2*fun1(grad1(xa));

[alpha_str, history_1d_nm] = newton1d(0.1, phi, dphi, ddphi, 1)
x_str  = xa -alpha_str*grad1(xa)

%% Problem 7: Function plotting
fun2_plot = @(x1,x2) (x2-x1).^4 +12*x1.*x2 -x1 +x2 -3;
a = -1.2; b = 1.2;

x = linspace(a,b,10);
y = linspace(a,b,10);
[X, Y] = meshgrid(x,y);
Z = fun2_plot(X,Y);

num_contours = 10;
% Plots
fig1 = figure(1);
surfc(X,Y,Z);
xlabel('X'); ylabel('Y'); zlabel('f');
title('Surface plot')
xlim([a,b]);
ylim([a,b]);
xticks(a:0.4:b);
yticks(a:0.4:b);
box('on');
view(3);
saveas(fig1,strcat(save_dir,'surf_plot_7a'),'epsc')

fig2 = figure(2);
contour(X,Y,Z,num_contours);
xlabel('X'); ylabel('Y');
title('Contour plot')
xlim([a,b]);
ylim([a,b]);
xticks(a:0.4:b);
yticks(a:0.4:b);
box('on');
saveas(fig2,strcat(save_dir,'contour_plot_7b'),'epsc')
%% Problem 8: Steepest Descent
fun2 = @(x) (x(2)-x(1)).^4 +12*x(1).*x(2) -x(1) +x(2) -3;
grad2 = @(x) [ -4*(x(2)-x(1)).^3 + 12*x(2) - 1; ...
               4*(x(2)-x(1)).^3 + 12*x(1) + 1];
x0_1 = [0.55; 0.7]; x0_2 = [-0.9; -0.5];
[x_str1_sd, history1_sd] = steepest_descent(x0_1, fun2, grad2);
x_str1_sd
[x_str2_sd, history2_sd] = steepest_descent(x0_2, fun2, grad2); 
x_str2_sd
% plotting Steepest descent paths
fig3 = figure(3);
hold on;
lvl_list1_sd = plot_traj(history1_sd,'red','o','x^0_1', fun2);
lvl_list2_sd = plot_traj(history2_sd, 'black','*','x^0_2', fun2);
lvl_list_sd = make_lvl_set(lvl_list1_sd,lvl_list2_sd);
contour(X,Y,Z,lvl_list_sd);
hold off;
xlabel('X'); ylabel('Y');
title('Steepest Descent Trajectory')
xlim([a,b]);
ylim([a,b]);
xticks(a:0.4:b);
yticks(a:0.4:b);
box('on');
saveas(fig3,strcat(save_dir,'plot_8_sd'),'epsc')

alpha_fixed = 0.05;
[x_str1_gd, history1_gd] = steepest_descent(x0_1, fun2, grad2, alpha_fixed);
x_str1_gd
[x_str2_gd, history2_gd] = steepest_descent(x0_2, fun2, grad2, alpha_fixed);
x_str2_gd

% plotting gradient descent paths
fig4 = figure(4);
hold on;
lvl_list1_gd = plot_traj(history1_gd,'red','o','x^0_1',fun2);
lvl_list2_gd = plot_traj(history2_gd, 'black','*','x^0_2', fun2);
lvl_list_gd = make_lvl_set(lvl_list1_gd, lvl_list2_gd);
contour(X,Y,Z,lvl_list_gd);
hold off;
xlabel('X'); ylabel('Y');
title('Gradient Descent Trajectory')
xlim([a,b]);
ylim([a,b]);
xticks(a:0.4:b);
yticks(a:0.4:b);
box('on');
saveas(fig4,strcat(save_dir,'plot_8_gd'),'epsc')

%% Problem 9: Newtons Method

Hessian2 = @(x) [ 12*(x(2)-x(1)).^2,-12*(x(2)-x(1)).^2+12 ;
                  -12*(x(2)-x(1)).^2+12, 12*(x(2)-x(1)).^3 ];
[x_str1_nm, history1_nm] = newton(x0_1, fun2, grad2, Hessian2, 1);
x_str1_nm
[x_str2_nm, history2_nm] = newton(x0_2, fun2, grad2, Hessian2, 1);
x_str2_nm
%plot Newtons method
fig5 = figure(5);
hold on;
lvl_list1_nm = plot_traj(history1_nm,'red','o','x^0_1', fun2);
lvl_list2_nm = plot_traj(history2_nm, 'black','*','x^0_2', fun2);
lvl_list_nm = make_lvl_set(lvl_list1_nm, lvl_list2_nm);
contour(X,Y,Z, lvl_list_nm);
hold off;
xlabel('X'); ylabel('Y');
title('Newtons Method Trajectory')
xlim([a,b]);
ylim([a,b]);
xticks(a:0.4:b);
yticks(a:0.4:b);
box('on');
saveas(fig5, strcat(save_dir,'plot_8_nm'),'epsc')

%% Local plotting function
function lvl_list = plot_traj(history,color,marker, disp_text, fun)
    lvl_list = [];
    for i=1:history.Niters -1
        x = history.data(i).x(1);
        y = history.data(i).x(2);
        u = history.data(i+1).x(1);
        v = history.data(i+1).x(2);
        plot([x,u],[y,v],'color',color,'marker',marker); 
        lvl_list = [lvl_list, fun([x,y])];
    end
    lvl_list = [lvl_list, fun([u,v])];
    text(history.data(1).x(1),history.data(1).x(2),disp_text, 'color',color, 'FontSize',15);
    
end
function L = make_lvl_set(list1, list2)    
    Lmin = min(min(list1), min(list2));
    Lmax = max(max(list1), max(list2));
    temp = linspace(Lmin,Lmax,10);
    L = [list1, list2, temp];
end