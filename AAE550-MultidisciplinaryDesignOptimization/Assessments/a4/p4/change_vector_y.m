%  function to find change vector y
function y=change_vector_y(x1,x0,lam)

% evaluations at x1
[f1,grad_f1]=fun(x1);
[g1,h1,grad_g1,grad_h1]=con(x1);

% evaluations at x0
[f,grad_f]=fun(x0);
[g,h,grad_g,grad_h]=con(x0);

y= grad_f1+grad_g1*lam ;% first term
y=y-( grad_f +grad_g*lam ); % second term
end