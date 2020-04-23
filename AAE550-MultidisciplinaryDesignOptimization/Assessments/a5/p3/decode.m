function val=decode(A,encoding)
% decodes the value of the given string using encoding containers

% find encoding length
b=keys(encoding);
b=length(b{1});

% split A into different values
l= length(A); % length of chromosome
n=l/b; 

val=[];
for i=1:n
   temp=A(b*(i-1)+1:b*i);
   temp=encoding(temp);
   val=[val,temp];    
end

end