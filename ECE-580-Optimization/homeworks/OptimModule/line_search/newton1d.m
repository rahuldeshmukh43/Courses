function [x_str, history_out] = newton1d(x_k,g,dg,ddg,verbose, TolFun, TolX, TolGrad)
    history.name = 'newton 1d';
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
    
    done = 0;
    count = 1;
    if verbose
       fprintf('Iter(k)\t x_k\n');
       fprintf('%d\t %0.4f\n',count, x_k); 
    end
    while ~done
        x_str = x_k - dg(x_k)/ddg(x_k);
        if abs(dg(x_str)) < TolGrad || abs(g(x_str) -g(x_k)) < TolFun || abs(x_str-x_k) < TolX
            done=1;
        end
        count = count +1;
        if verbose
           fprintf('%d\t %0.4f\n',count, x_str); 
        end        
        x_k = x_str;
        history.data(count).x = x_k;         
    end
    history.Niters = count-1;    
    if nargout>1
       history_out = history; 
    end    
end