function [x_str, fval] = mylinprog(c,A,b,Aeq,beq,LB,UB,verbose)
% Two Phase Simplex LP solver
% the problem is not given in std form
% Given: min c'*x
% st:    Aeq*x = beq
%        Ax <= b
%        0 <= x  or LB <= x <= UB
if nargin==7
    verbose=0; %by default
end
% transform variables: z = x - LB if LB is not 0
if ~isempty(LB) 
%     error('Not implemented');
   b = b - A*LB;
   if ~isempty(Aeq)
       beq = beq - Aeq*LB;
   end
end
% stack ineqaulity conditions Iz <= UB-LB
if ~isempty(UB)
    A = [A;eye(length(UB))];
    b = [b; UB-LB];
end
%% Convert to std form using slack and surplus variables st b_std>=0
A_std =[]; b_std = []; c_std =[];
A_std = [A_std; A, eye(size(A,1))]; 
b_std = [b_std;abs(b)];
c_std = [c_std;c;zeros(size(A,1),1)];
% make inequality b's positive
neg_b_idx = find(b < 0);
A_std(neg_b_idx,:) = -1*A_std(neg_b_idx,:);
% stack equality equations
if ~isempty(Aeq)
    neg_b_idx = find(beq < 0);
    Aeq(neg_b_idx,:) = -1*Aeq(neg_b_idx,:);
    A_std = [A_std; Aeq, zeros(size(Aeq,1),size(A,1))];
    b_std = [b_std; abs(beq)];
end    
%% Solve std form using two-phase simplex method
if verbose
    fprintf('\\text{Phase 1:}&\\nonumber\\\\\n%%\n');
end
% Phase1: Find initial basis using artificial variables
A1 = [A_std, eye(size(A_std,1))];
c1 = [zeros(size(A_std,2),1);ones(size(A_std,1),1)];
basis_idx = size(A_std,2) + (1:size(A_std,1));
[~, ~, basis_idx_p1, tab]= simplex(A1,b_std,c1, basis_idx, verbose);
if verbose
   fprintf('\\text{Phase 2:}&\\nonumber\\\\\n%%\n') ;
end
% Phase2: Find optimal solution
A2 = tab(1:end-1,1:size(A_std,2));
b2 = tab(1:end-1,end);
[x_str_p2, fval,~, ~] = simplex(A2, b2, c_std, basis_idx_p1, verbose);
x_str = x_str_p2(1:length(c));
if ~isempty(LB)
    % transform from z_str to x_str, x_str= z_str + LB
   x_str = x_str + LB; 
   fval = fval + c'*LB;
end
fprintf('\n** Optimum Solution found using mylinprog **\n');
if verbose
    display(x_str);
    display(fval);
end
end
