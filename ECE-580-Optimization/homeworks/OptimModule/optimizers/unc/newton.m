function [x_str, history_out] = newton(x0, fun, grad, Hess, ...
                        verbose, ...
                        TolFun, TolX, TolGrad)
    history.name = 'Newtons Method';
    switch nargin 
        case 4
            verbose = 0;
            TolFun = 1e-6;
            TolX = 1e-6;
            TolGrad = 1e-10;
        case 5
            TolFun = 1e-6;
            TolX = 1e-6;
            TolGrad = 1e-10;
        case 6
            TolX = 1e-6;
            TolGrad = 1e-10;
        case 7
            TolGrad = 1e-10;
    end
    
    history.parameter.TolFun = TolFun;
    history.parameter.TolX = TolX;
    history.parameter.TolGrad = TolGrad;
    
    if verbose
        fprintf('Iter\t fun_val\n')
    end    
    x_str = x0;
    done = 0;
    f_str = fun(x_str);
    grad_str = grad(x_str);    
    iter = 0;
    history.data(iter+1).x = x_str;
    while ~done
        iter = iter +1;
        x_prev = x_str;
        f_prev = f_str;
        x_str = x_prev - inv(Hess(x_prev))*grad_str;
        f_str = fun(x_str);
        grad_str = grad(x_str);
        
        fun_val_crit = abs(f_str - f_prev) < TolFun; 
        x_norm_crit = sum((x_str-x_prev).^2) < TolX^2;
        grad_crit = sum(grad_str.^2) < TolGrad^2; 
        %print statement
        if verbose
            fprintf('%d\t %0.4f\n', iter, f_str);
            if iter > 100
                fprintf('Niter limit crossed no optim solution\n');
                break;
            end
        end
        if fun_val_crit || x_norm_crit ||grad_crit
            if fun_val_crit && verbose
                fprintf('TolFun criteria satisfied\n');
            end
            if x_norm_crit && verbose
                fprintf('TolX criteria satisfied\n');
            end
            if grad_crit && verbose
                fprintf('TolGrad criteria satisfied\n');
            end
            done = 1;            
        end
        history.data(iter+1).x = x_str;
    end
    history.Niters = iter;
    
    if nargout>1
        history_out = history;
    end        
end