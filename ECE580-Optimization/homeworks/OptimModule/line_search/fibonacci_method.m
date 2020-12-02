function [xa, xb, history_out] = fibonacci_method(xa, xb, fun, TolX, verbose)
    history.name = 'Fibonacci Method';
    delta = 0.1; % for last value of rho
	switch nargin
		case 3
			TolX= 1e-6;
			verbose = 0;
		case 4
			verbose = 0;
    end
    history.params.delta = delta;
    history.params.TolX = TolX;
    % estimate Niter
    Niter = get_Niter_fibb(xa,xb,delta, TolX);
    history.Niters = Niter;
    
    % do range reduction
    k = 1;
    rho_k = 1 - (fibonacci(Niter+1)/fibonacci(Niter+1+1));
    s = xa + rho_k*(xb-xa);
	t = xa + (1-rho_k)*(xb-xa);
	f1 = fun(s);
	f2 = fun(t);
    f_xa = fun(xa);
    f_xb = fun(xb);
    %
    if verbose
        %fprintf('\\hline\nIter(k)&\t rho_k&\t ak&\t bk&\t f(ak)&\t
        %f(bk)&\t New uncertainity interval (a,b)&\t Uncertainity
        %width\\\\\n\\hline\n'); for latex
        fprintf('Iter(k)\t rho_k\t ak\t bk\t f(ak)\t f(bk)\t New uncertainity interval (a,b)\t Uncertainity width\n');
        rowvec_fmt = ['[', repmat('%0.4f, ', 1, numel(xa)-1), '%0.4f]']; 
    end
    
    for k=1:Niter
        xa_prev = xa;
        xb_prev = xb;
        f_xa_prev = f_xa;
        f_xb_prev = f_xb;
        rho_k = 1 -(fibonacci(Niter-(k-1)+1)/fibonacci(Niter+1 -(k-1)+1));        
        if k == Niter
           rho_k = rho_k - delta; 
        end
        if f1 < f2
			xb=t;
            f_xb = f2;
			t=s;
			s= xa + rho_k*(xb-xa); 			
            f2 = f1;
			f1 = fun(s);            
		else 
			xa = s;
            f_xa = f1;
			s = t;
			t = xa + (1-rho_k)*(xb-xa);
			f1 = f2;
			f2 = fun(t);
        end
        %
         if verbose
%             fprintf('%d&\t',k);
%             fprintf('%0.4f&\t',rho_k);
%             fprintf(rowvec_fmt,xa_prev); fprintf('&\t');
%             fprintf(rowvec_fmt,xb_prev); fprintf('&\t');
%             fprintf('%0.4f&\t %0.4f&\t',f_xa_prev,f_xb_prev);
            fprintf('%d\t',k);
            fprintf('%0.4f\t',rho_k);
            fprintf(rowvec_fmt,xa_prev); fprintf('\t');
            fprintf(rowvec_fmt,xb_prev); fprintf('\t');
            fprintf('%0.4f\t %0.4f\t',f_xa_prev,f_xb_prev);
            fprintf('(');
            fprintf(rowvec_fmt,xa);
            fprintf(' , ')
            fprintf(rowvec_fmt,xb);
            %fprintf(')&\t%0.4f\\\\\n\\hline\n',norm((xb-xa),2)); %latex
            fprintf(')\t%0.4f\n',norm((xb-xa),2));
        end
        %
        history.data(k).x_prev = [xa_prev,xb_prev];
        history.data(k).x_new = [xa,xb];
    end
    if verbose
        fprintf("** Fibonacci method took %d iters **\n** Final uncertainity region width was %0.4f **\n",...
            Niter,norm((xb-xa),2));
    end
    
    if nargout>2
       history_out = history; 
    end
    
end

function N = get_Niter_fibb(xa,xb,delta,TolX)
    N = 1;
    d = norm((xb-xa),2);
    while d*(1+2*delta)/TolX > fibonacci(N+1+1)
        N = N +1;
    end
    
end

