function [x_k, history_out] = steepest_descent(x_k,fun,grad,alpha,...
                                verbose, TolFun, TolX, TolGrad)
    switch nargin
        case 3
            alpha_fixed = 0;
            verbose = 0;
            TolFun = 1e-4;
            TolX = 1e-4;
            TolGrad = 1e-4;
        case 4
            alpha_fixed = 1;
            verbose = 0;
            TolFun = 1e-4;
            TolX = 1e-4;
            TolGrad = 1e-4;
        case 5
            alpha_fixed = 1;
            TolFun = 1e-4;
            TolX = 1e-4;
            TolGrad = 1e-4;
        case 6
            alpha_fixed = 1;
            TolX = 1e-4;
            TolGrad = 1e-4;
        case 7
            alpha_fixed = 1;
            TolGrad = 1e-4;
    end
    
    history.parameter.TolFun = TolFun;
    history.parameter.TolX = TolX;
    history.parameter.TolGrad = TolGrad;
    
    done = 0;    
    if alpha_fixed
        history.name = 'Grad Descent (fixed alpha)';
    else
        history.name = 'Steepest Descent';
    end    
    count=1;
    history.data(count).x = x_k;
    f_k = fun(x_k);
    history.data(count).f_k = f_k;
    g_k = grad(x_k);
    history.data(count).g_k = g_k;
    while ~done
        d_k = -g_k;
        if ~alpha_fixed
            [xa,xb, initial_interval_summ]= get_search_interval(x_k, fun, d_k);
            history.data(count).initial_search_interval_summary = initial_interval_summ;
            [xa,~,line_search_summary] = fibonacci_method(xa, xb, fun);
            history.data(count).line_search_summary = line_search_summary;
            alpha_k = norm((xa-x_k),2)/norm(d_k,2);
        end
        history.data(count).alpha_k = alpha_k;
        delta_x_k = alpha_k*d_k;
        x_k = x_k + delta_x_k;
        g_k = grad(x_k);
        f_k_plus_1 = fun(x_k);
        
        count = count+1;
        history.data(count).x = x_k;
        history.data(count).f_k = f_k_plus_1;
        history.data(count).g_k = g_k;
        
        if abs(f_k_plus_1 -f_k) < TolFun || norm(alpha_k*d_k,2) <TolX || norm(g_k)< TolGrad
            done = 1;
        end        
    end
    history.Niters = count -1;
    
    if nargout>1
        history_out = history;
    end
    f_k = f_k_plus_1;
end