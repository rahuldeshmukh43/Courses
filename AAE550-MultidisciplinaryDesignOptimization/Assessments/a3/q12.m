%assessment 3 Q12 Penalty methods
clc; clear all; close all;
%problem definition
f=@(x) 6*x(1)^2+13*x(2)^2-3*x(1)*x(2);

g1=@(x) (1/8)*x(1)^2+(1/6)*x(2)^2-1;
g2=@(x) 2-x(1);
g3=@(x) -x(2);
g_vec={g1,g2,g3};

%part A
x0=[0 1]';
rp=1;
phi_qep = pseudo_obj_qep(f,g_vec,x0,rp);%17
display(phi_qep);

%part B
rp_ip=10;% for interior penalty method
phi_ipc = pseudo_obj_ip(f,g_vec,x0,rp_ip);%30
display(phi_ipc);

%part C: ps obj classical Linear extended interior penalty
ep=-0.1;
phi_ip_linearex=pseudo_obj_ip_linearEx(f,g_vec,x0,rp_ip,ep);
display(phi_ip_linearex);

%partD: clasical quadratic extended interior penalty
phi_ip_quadex=pseudo_obj_ip_quadEx(f,g_vec,x0,rp_ip,ep);
display(phi_ip_quadex);


%part E