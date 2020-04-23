function [xa, xb, history_out]= golden_section(xa,xb,fun,TolX,verbose)
	history.name = 'Golden Section';
    switch nargin
		case 3
			TolX= 1e-6;
			verbose = 0;
		case 4
			verbose = 0;
    end
    
	rho= 1 - (sqrt(5)-1)/2;
	s = xa + rho*(xb-xa);
	t = xa + (1-rho)*(xb-xa);
	f1 = fun(s);
	f2 = fun(t);
    f_xa = fun(xa);
    f_xb = fun(xb);
	% compute Niter
	Niter = floor(log(TolX/(sqrt(sum((xb-xa).^2))))/log(1-rho)) + 1 ;
    history.Niters = Niter;
    history.parameter.rho = rho;
    %
    if verbose
        fprintf('\\hline\nIter(k)&\t ak&\t bk&\t f(ak)&\t f(bk)&\t New uncertainity interval (a,b)&\t uncertainity width\\\\\n\\hline\n');
        rowvec_fmt = ['[', repmat('%0.4f, ', 1, numel(xa)-1), '%0.4f]']; 
    end
    %     
	for i=1:Niter
        xa_prev = xa;
        xb_prev = xb;
        f_xa_prev = f_xa;
        f_xb_prev = f_xb;
        if f1 < f2
			xb=t;
            f_xb= f2;
			t=s;
			s= xa + rho*(xb-xa); 
			f2 = f1;
			f1 = fun(s);
		else 
			xa = s;
            f_xa = f1;
			s = t;
			t = xa + (1-rho)*(xb-xa);
			f1 = f2;
			f2 = fun(t);
        end
        %
        if verbose
            fprintf('%d&\t',i);
            fprintf(rowvec_fmt,xa_prev); fprintf('&\t');
            fprintf(rowvec_fmt,xb_prev); fprintf('&\t');
            fprintf('%0.4f&\t %0.4f&\t',f_xa_prev,f_xb_prev);
            fprintf('(');
            fprintf(rowvec_fmt,xa);
            fprintf(' , ')
            fprintf(rowvec_fmt,xb);
            fprintf(')&\t %0.4f\\\\\n\\hline\n',sqrt(sum((xb-xa).^2)));
        end
        history.data(i).x_prev = [xa_prev, xb_prev];
        history.data(i).x_new = [xa,xb];
	end
	fprintf("** Golden section took %d iters **\n** Final uncertainity region width was %0.4f **\n",...
        Niter,sqrt(sum((xb-xa).^2)));	
    if nargout > 2
        history_out = history;
    end
end
