function [x_prev, x_nxt, history] = get_search_interval(x0, fun, direction,...
                                                step_size)
% function for identifying the interval which contains a minimizer
% using x0 as initial guess and evaluating the fun till the function value
% increases btw 2 consecutive evals
% Input: fun: function handle for obj fun
%        step_size: initial step size, user param
%        direction: function handle for search direction
    switch nargin
        case 3
            step_size = 0.1;
    end        
    x_prev = x0;    
    x_cur = x_prev + step_size*direction;
    step_size = 2*step_size;
    x_nxt = x_cur + step_size*direction;
    
    f1 = fun(x_prev);
    f2 = fun(x_cur);
    f3 = fun(x_nxt);
    f = [f1,f2,f3];
    
    count = 1;
    history.data(count).interval = [x_prev, x_nxt];
    while f(2) >= f(3)
        step_size = 2*step_size;
        x_prev = x_cur;
        x_cur = x_nxt;
        x_nxt = x_cur + step_size*direction;
        f = [f(2),f(3), fun(x_nxt)];
        count = count +1;
        history.data(count).interval = [x_prev, x_nxt];
    end
    if nargout>2
        history.name = 'initial search interval';
        history.Niters = count;
        history.parameter.step_size = step_size;
    end
end
    
    