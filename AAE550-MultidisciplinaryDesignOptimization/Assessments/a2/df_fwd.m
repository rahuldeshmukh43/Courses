function df_fwd=df_fwd(f,x,delta_x)
% function for finding gradient at a point using fwd difference
%function f is a scalar funciton and can take x= array as input
% x is a column vector
%function made for Q10 of assessment 2
df_fwd=zeros(size(x));
temp=zeros(size(x));
for i=1:length(x)
   temp(i)=delta_x;
   df_fwd(i)=(f(x+temp)-f(x))/delta_x;
   temp(i)=0;
end
end