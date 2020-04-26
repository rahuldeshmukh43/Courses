function [x_str, fval, basis_idx, tab]= simplex(A,b,c, basis_idx, verbose)
% simplex method to solve LP in std form:
% Given: min c^Tx
% st:    Ax = b      
%        x>=0
% basis_idx the indices are in order of std cartesian basis 1...n
tol = 1e-6;
tab = [A, b;
       c',0];
if verbose
    print_tab(tab);
end

% make cost coeffs zero for basis idx
for i=1:length(basis_idx)
%     r = find(tab(1:end-1,basis_idx(i))>0);
    tab(end,:) = tab(end,:) - tab(end,basis_idx(i))*tab(i,:);
end

% till all cost coeffs are non-negative do:
while ~isempty(find(tab(end,1:end-1)< 0))    
% choose pivot column
    [~,p] = min(tab(end,1:end-1));
% find pivot element
    % using only positive a_p's 
    pos_ap_idx = find(tab(1:end-1,p)>0);
    [~,q_idx] = min(tab(pos_ap_idx,end)./tab(pos_ap_idx, p));
    q = pos_ap_idx(q_idx);
    % update basis idx
    basis_idx(q) = p;
% print tab with pivot element
    if verbose
        print_tab(tab,[q,p]);
    end
    % make pivot 1
    tab(q,:) = tab(q,:)/tab(q,p);
% make pivot column
    for r=1:size(tab,1)
        if r~=q
            tab(r,:) = tab(r,:) - tab(r,p)*tab(q,:);
        end
    end
    tab(tab>-tol & tab<tol) = 0;
end
if verbose
    print_tab(tab)
end
%return x_str and basis_idx
x_str = zeros(length(c),1);
x_str(basis_idx) = tab(1:end-1,end);
fval= c'*x_str;
end

function print_tab(tab, pivot)
if nargin==1
    pivot = 0;
end
% display(tab)
% print latex bmatrix 
[n_row, n_col] = size(tab);
fprintf('&=\\begin{bmatrix}\n');
for r=1:n_row
    for c=1:n_col-1
        if r==pivot(1) && c== pivot(2)
            % this is the pivot
            fprintf('\\fbox{{\\color{red}%s}}&\t',strtrim(rats(tab(r,c))));  
        else
            fprintf('%s&\t',strtrim( rats(tab(r,c)) ));        
        end
    end
    fprintf('%s\\\\\n',strtrim( rats(tab(r,c+1)) ));
end
fprintf('\\end{bmatrix}\\nonumber\n%%\n');
end