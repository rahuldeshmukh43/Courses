function [x_k, history_out] = CG(x_k,fun,grad,...
                                verbose, TolFun, TolX, TolGrad)
    history.name = 'Conjugate Gradient';
    switch nargin
        case 3
            verbose = 0;
            TolFun = 1e-4;
            TolX = 1e-4;
            TolGrad = 1e-4;
        case 4            
            TolFun = 1e-4;
            TolX = 1e-4;
            TolGrad = 1e-4;
        case 5            
            TolX = 1e-4;
            TolGrad = 1e-4;
        case 6
            TolGrad = 1e-4;
    end
    history.parameter.TolFun = TolFun;
    history.parameter.TolX = TolX;
    history.parameter.TolGrad = TolGrad;
    
    done = 0;
    count=1;
    f_k = fun(x_k);
    g_k = grad(x_k);
    history.data(count).x = x_k;
    history.data(count).f_k = f_k;    
    history.data(count).g_k = g_k;
    d_k = -g_k;
    while ~done
        % get the step length
        % initial search interval
        [xa,xb, initial_interval_summ]= get_search_interval(x_k, fun, d_k);
        history.data(count).initial_search_interval_summary = initial_interval_summ;
        % 1-d line search using fibonacci method
        [xa,~,line_search_summary] = fibonacci_method(xa, xb, fun);
        history.data(count).line_search_summary = line_search_summary;
        alpha_k = norm((xa-x_k),2)/norm(d_k,2);
        history.data(count).alpha_k = alpha_k;
        
        delta_x_k = alpha_k*d_k;
        x_k = x_k + delta_x_k;
        delta_g_k = grad(x_k) -g_k;
        old_g_k_norm  = g_k'*g_k;
        g_k = g_k + delta_g_k;
        
        if mod(count,length(x_k)+1) ~= 0
            beta_k = max(0, (g_k'*(delta_g_k))/old_g_k_norm);
            d_k = -g_k + beta_k*d_k;
        else
            % reset to neg grad
            d_k = -g_k;
        end
        
        count = count+1;
        f_k_plus_1 = fun(x_k);
        history.data(count).x = x_k;
        history.data(count).f_k = f_k_plus_1;
        history.data(count).g_k = g_k;
        history.data(count).alpha_k = alpha_k;
        
        % check if done
        if abs(f_k_plus_1 - f_k) < TolFun || norm(delta_x_k ,2) <TolX || norm(g_k,2)< TolGrad || count>100
            done = 1;
        end
        f_k = f_k_plus_1;
    end    
    history.Niters = count -1;    
    if nargout>1
        history_out = history;
    end
end