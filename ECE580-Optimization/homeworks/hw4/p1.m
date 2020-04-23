clc; clear all;
format short;
rng('default')

a_1 = rand(2,1)
a_2 = rand(2,1)
A_1 = a_1*a_1'
A_2 = a_2*a_2'

A_1_pinv = pinv(A_1);
A_2_pinv =pinv(A_2);

A_2_pinv_A_1_pinv = A_2_pinv*A_1_pinv
A_1A_2_pinv = pinv(A_1*A_2)

diff = A_1A_2_pinv - A_2_pinv_A_1_pinv

if abs(sum(diff))> 1e-16
    fprintf('pinv(A_2)pinv(A_1) not equals pinv(A_1A_2)\n')
end