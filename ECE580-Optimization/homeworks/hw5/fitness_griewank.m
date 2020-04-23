function f = fitness_griewank(X_coded, lb, code_lens, resolution)
X = decode(X_coded, lb, code_lens, resolution);
f = griewank_fun(X');
end