% ECE580 MidTerm-2
% Rahul Deshmukh
% deshmuk5@purdue.edu
format short;
addpath('../OptimModule/line_search/');
addpath('../OptimModule/optimizers/unc/');
addpath('../OptimModule/optimizers/unc/QuasiNewton/');
%% Problem -A
fprintf('----P1-------\n');
f = @(x) x(1)^2 + 0.5*x(2)^2 -x(1) +x(2) +7;
g = @(x) [2*x(1) - 1; x(2) + 1];
Q = [2,0;0,1];
x0 = [0;0];
% [x_str, history] = DFP(x0, f, g)
g0 = g(x0);
H0 =eye(2);
d0 = -1*H0*g0;
alpha0 = -1*(g0'*d0)/(d0'*Q*d0)
x1 = x0+ alpha0*d0;
delta_x = x1-x0;
g1 = g(x1);
delta_g = g1-g0;
%
v_k = (delta_g'*delta_x);
delta_H_k1 = (delta_x*delta_x')/v_k;
delta_H_k2 = ((H0*delta_g)*(H0*delta_g)')/(delta_g'*H0*delta_g);
H1 = H0 + delta_H_k1 - delta_H_k2;
%
d1 = -1*H1*g1;
alpha1 = -1*(g1'*d1)/(d1'*Q*d1)
% alpha1 =
% 
%     0.8333
fprintf('-----------------\n');
%% Problem -B
fprintf('----P2-------\n');
A = [eye(2);1,1];
b = [1;1;0];
x_str = pinv(A)*b;
x_str(1)
% x_str =
% 
%     0.3333
%     0.3333
fprintf('-----------------\n');
%% Problem - C
fprintf('----P3-------\n');
A = [1,0;0,1;0,-1];
b = [0;2;1] - [-2;0;0];
x_str = pinv(A)*b;
x_str(1)
% x_str =
% 
%     2.0000
%     0.5000
fprintf('-----------------\n');
%% Problem - D
fprintf('----P4-------\n');
lhs = [1 0 -1;
       0 2 1;
       0 -1 0];   
Pb = [0;-1;1];
cP = [0; 1; 1];
cPb = 0;
M = lhs + (Pb*cP')/(1+ cPb);
M(1,3)
% ans =
% 
%     -1
fprintf('-----------------\n');
%% Problem - E
fprintf('----P5-------\n');
A0 = [0 1;
      1 1;
      0 1];
b0 = [0;1;0] ;
a1 = [1;0];
b1 = 2;
P0 = inv(A0'*A0);
x0 = P0*A0'*b0;
P1 = P0 - (P0*a1*a1'*P0)/(1+ a1'*P0*a1);
x1 = x0 + P1*a1*(b1 - a1'*x0);
P1(2,2)
% ans =
% 
%     0.4000
fprintf('-----------------\n');
%% Problem - F
fprintf('----P6-------\n');
A0 = [0 1;
      1 1;
      0 1];
A1 = [1 0];
A = [A0;A1];
pinvA = pinv(A);
pinvA(2,4)
fprintf('-----------------\n');
%% Problem - G
fprintf('----P7-------\n');
A = [0 0;
     1 2;
     1 2];
b = [2; 1; 2];
x_str = pinv(A)*b
x_str(2)
% x_str =
% 
%     0.3000
%     0.6000
fprintf('-----------------\n');
%% Problem - H
fprintf('----P8-------\n');
x_cur = [1;2];
v_cur = [0.5; 3.5];
p = [0.5; 1.5];
g = [5;6];
r = 0.5*[1;1];
s = 0.25*[1;1];

omega = 1; c1 = 2; c2 = 2;
phi = c1+ c2;
kappa = 2/abs(2- phi - sqrt(phi^2 -4*phi))

v_nxt = kappa*(omega*v_cur + c1*r.*(p-x_cur) + c2*s.*(g-x_cur));
x_nxt = x_cur + v_nxt;
x_nxt(2)
% ans =
% 
%      7
fprintf('-----------------\n');
%% Problem - I
fprintf('----P9-------\n');
pm = 0.1; pc = 0.5;
f = 12; F=5; e= 10;
l_H= 4-2;
L = 7;
O_H = 2;
bound= ((f/F)*e)*(1 - pc*(l_H./(L-1))).*((1-pm).^O_H)

fprintf('-----------------\n');
%% Problem - J
fprintf('----P10-------\n');
c= [3;1;1];
Aeq = [1 0 1;
        0 1 -1];
beq = [4;2];
A = []; b = [];
[x_str,fval] = linprog(c, A, b, Aeq, beq, [0;0;0]);
x_str(1)

% ans =
% 
%      0
