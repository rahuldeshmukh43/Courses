function [g,h,grad_g,grad_h]=NonLinCon(x,a_g)
% function for generating non linear constraint for SQP
% input: x: col vector of desing variables
% a_g : matrix of constraints with coefficients stacked as col vectors

% find number of design variables
n = length(x);
% make the row for of basis vector
combinations = combnk(1:n,2);
xixj = [];
for i=1:size(combinations,1)
    c1= combinations(i,1);
    c2= combinations(i,2);
    xixj = [xixj;x(c1)*x(c2)];
end

X = [1; x; xixj; x.^2];

g = a_g'*X -1; 

h=[]; % equality constriant

if nargout > 2
    grad_g = [];% to be stacked as cols
    for ig =1:size(a_g,2)
        % for igth constraint
        grad_gi = [];
        for i=1:n
            % for a0 term
            igrad = [0] ;
            % for aixi terms
            temp = zeros(n,1);
            temp(i) = 1;
            igrad = [igrad;temp];
            % for aijxixj terms
            temp = zeros(size(combinations,1),1);
            for j =1:size(combinations,1)
                if ~isempty(find(combinations(j,:)==i))
                    temp_copy =  combinations(j,:);
                    temp_copy(find(combinations(j,:)==i))='';
                    temp(j) = x(temp_copy);
                end
            end
            igrad = [igrad;temp];
            % for xi^2 terms
            temp = zeros(n,1);
            temp(i) = 2*x(i);
            igrad = [igrad;temp];
            grad_gi = [grad_gi; a_g(:,ig)'*igrad];
        end
        grad_g = [grad_g,grad_gi];
    end
    grad_h = [];
end
end