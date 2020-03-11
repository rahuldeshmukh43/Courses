% assessment 5 P2: Simulated Annealing V15
clc; clear all;

format short;
% define the function 
f=@(x) -3*sin(0.95*norm(x)^3)/norm(x);

n=4;
lb=-12*ones(n,1);
ub= 12*ones(n,1);

x0=[4;-1;4;-2];
v0=4.5*ones(n,1);
e=eye(n,n); %e_i

T0=1;

% list of random numbers
ran_num=[-0.8608  0.1922  0.0360  0.7873 -0.2099 -0.0203  0.1269 -0.0966  0.3120 -0.3270];
ran_num=kron(ones(1,100),ran_num);

% loop for n directions 
%initialize 
x_star=x0;
f_star=f(x0);
% start cycles loop
for i=1:n
    fprintf(strcat('i = ',num2str(i)));
    % update in ith direction
    r=ran_num(1); ran_num=ran_num(2:end);

    x_prime=x0+r*v0(i)*e(:,i)
    % find change in f
    del_f=f(x_prime)-f(x0)
    
    % Metropolis criterion: for acceptability of x_prime
    if del_f<=0
        x0=x_prime;
        % update best solution ever 
        x_star=x0;
        f_star=f(x0);
        fprintf('x_prime is Acceptable: Downhill\n');
    else
        P=exp(-del_f/T0)
        p_prime=abs(ran_num(1)); ran_num=ran_num(2:end);
        if p_prime<=P
            x0=x_prime;
            fprintf('x_prime acceptable Uphill\n');
        else
            fprintf('x_prime is unacceptable: stay at current point\n');
        end
    end 
    fprintf('-----------------------------------------\n');
end

% final function value at the end of one loop
f_star
