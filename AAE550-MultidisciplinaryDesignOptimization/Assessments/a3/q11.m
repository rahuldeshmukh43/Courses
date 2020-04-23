%assessment 3 Q11 Conjugate Direction and Newtons method
clc;clear all; close all;format long;
%q1
% syms x1 x2;
% f_sym= 18*x1^4 -4*x1^2*x2+3*x2^2+11*x1^2-2*x1+6;
% df_sym=gradient(f_sym,[x1,x2]);
% h_sym=hessian(f_sym,[x1,x2]);

f=@(x) 18*x(1)^4 -4*x(1)^2*x(2)+3*x(2)^2+11*x(1)^2-2*x(1)+6;
df=@(x)[ 22*x(1) - 8*x(1)*x(2) + 72*x(1)^3 - 2; -4*x(1)^2 + 6*x(2)];
H=@(x)[216*x(1)^2-8*x(2)+22,-8*x(1);-8*x(1),6];

%part A
x0=[2 2]';
x1=[0.2131;2.0122];
s1=-1*df(x0);
beta_1=(norm(df(x1))/norm(df(x0)))^2;
s2_C=-1*df(x1)+beta_1*s1; %s2_C=[  -0.195865500052928 -11.889906429313987]

%part B
% syms a; 
% vpa(simplify(f(x1+a*s2_C)))
f1_a_C=@(a) 0.026491362030824757514006432459974*a^4 + 1.7092544019151860649428834475073*a^3 + 420.44081690969846457214759738214*a^2 - 141.38055916663047892933455044551*a + 17.8917851212422578;
a2_C=fminbnd(f1_a_C,0,5);%   0.167970617319532

x2_C= x1+a2_C*s2_C;
display(x2_C);%   [0.180200351044511   0.015045077196662]


%part C



%part D
H_0=H(x0);
fprintf('H(x0)=');
display(H_0);%   [870   -16;   -16     6]

%part E
b=-1*df(x0);
s1_N=H_0\b;%search direction Newton
%find step length:a
% syms a;
% x1_sym=x0+a*s1_N;
% vpa(simplify(f(x1_sym)));
f1_a=@(a) 4.209481080932502777537910743212*a^4 - 46.128561096994447827532220890999*a^3 + 201.3787268331990330378726833199*a^2 - 402.75745366639806607574536663981*a + 314.0;
a1_N=fminbnd(f1_a,0,5);%   2.416227194994370
x1_N=x0+a1_N*s1_N;
fprintf('x1 Newton=');
display(x1_N);%   [0.319738864399564  -0.869878231604917]