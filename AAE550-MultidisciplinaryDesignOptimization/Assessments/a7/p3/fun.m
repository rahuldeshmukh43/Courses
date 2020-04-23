function [f,grad_f]= fun(x,a_m)
%  function for finding the mass for a particular x
% m: col vector of coefficients of the approximation 

n = length(x); %number of design variables

% generate the n dim quadratic row
%  st [1;xi,xi*xj;xi^2]
combinations = combnk(1:n,2);
xixj = [];
for i=1:size(combinations,1)
    c1= combinations(i,1);
    c2= combinations(i,2);
    xixj = [xixj;x(c1)*x(c2)];
end

X = [1; x; xixj; x.^2];

f = a_m'*X; 

if nargout > 1  
    grad_f = [];
    for i=1:n
       % for a0 term
       igrad = [0] ;
       % for aixi terms
       temp = zeros(n,1);
       temp(i) = 1;
       igrad = [igrad;temp];
       % for aijxixj terms
       temp = zeros(size(combinations,1),1);
       for j =1:size(combinations,1)
           if ~isempty(find(combinations(j,:)==i))
               temp_copy =  combinations(j,:);
               temp_copy(find(combinations(j,:)==i))='';
               temp(j) = x(temp_copy);
           end
       end
       igrad = [igrad;temp];
       % for xi^2 terms
       temp = zeros(n,1);
       temp(i) = 2*x(i);
       igrad = [igrad;temp];
       grad_f = [grad_f; a_m'*igrad];
    end
end

end