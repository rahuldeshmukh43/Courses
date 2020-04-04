function f = fitness(X_coded, lb, code_lens, resolution,...
                    x_pos, y_pos)
X = decode(X_coded, lb, code_lens, resolution);
[N_pop,~] = size(X);
f = zeros(N_pop,1);
for i=1:N_pop
   ith_route = X(i,:);
   f(i) = route_len(ith_route,x_pos,y_pos);
end
end

function d = route_len(r, x_pos, y_pos)
r_end = [r(2:end),r(1)];
delta_x = x_pos(r_end) - x_pos(r);
delta_y = y_pos(r_end) - y_pos(r);
d = sum(sqrt(delta_x.^2 + delta_y.^2));
end