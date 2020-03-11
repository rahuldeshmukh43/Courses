function f=fun(x)
% evaluates the function
% Input: x is a col vector
% Output: f is a scalar
f=0;
for i=1:length(x)
   temp= (ceil(abs(x(i)))^4)*floor(x(i));
   f=f+temp; 
end
    
end