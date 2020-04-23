% assessment 7: P1 Q26 Least Squares V01 
clc; clear all;
n = 5;
(n+1)*(n+2)/2

% Part B: size of X for n=5; and ndp = 3^n
ndp = 3^n
col = 1+ 2*nchoosek(n,1)+nchoosek(n,2)

% Part C: row of basis vector for
x = [2;4];
[1;x;x(1)*x(2);x.^2]'

fprintf('---------------------------------------------------------------\n');
% Q27 : Response surface basic concepts
n = 5;
% PArt A:
fprintf('FFD\n');
3^n  %trials

fprintf('CCD');
n=7;
2^n+2*n+1

fprintf(' Box n Behnken \n');
n=9;
size(bbdesign(n,'center',1))

fprintf(' Saturated D Optimal \n')
n=11;
(n+1)*(n+2)/2