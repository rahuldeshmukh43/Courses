% assessment 5 Nelder Mead simplex Q19
% rahul Deshmukh
% 24 Oct 2018
% deshmuk5@purdue.edu
%%
clc; clear all;
% function is defined in fun.m
x0=[4;-2;4;2];
v=3*ones(4,1);
a=1; % alpha

% find nmber of design variables
n=length(x0);

% find n+1 vertices
E=eye(n); % cols of E are e_i
% X vector of all x_i i=0,1,...n
% find x_i using x_i= x0+vi*ei
X=kron(x0,ones(1,n))+diag(v);
X=[x0,X] % x_i as col vectors

% evaluate function at all x_i
f=[]; % row vector of function values
for i=1:n+1
    f=[f,fun(X(:,i))];
end
f
% find max, second max, min
sorted_f=sort(f);% in increasing order
h=find(f==sorted_f(n+1)); % max
h=h(2);
s=find(f==sorted_f(n)); % second max
s=s(1);
l=find(f==sorted_f(1)); % min
x_h=X(:,h)
x_s=X(:,s)
x_l=X(:,l)

% centroid 
temp_X=X; temp_X(:,h)='';
c=(1/n)*(temp_X*ones(n,1))

% reflection -iter1
x_r=c+a*(c-x_h)

% evaluate f(x_r)
f_xr=fun(x_r)
% check if less than fun(x_l)
fprintf('Next Move is: ');
if f_xr<f(l)
    fprintf('Do expansion\n');
elseif f_xr<f(s)
    fprintf('accept reflection\n');
elseif f_xr>f(s)
    fprintf('do contraction\n');
end
