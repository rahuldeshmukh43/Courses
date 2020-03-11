function winner=tournament(A,B,encoding,f)
% carries out tournament btw A and B using fitness function f, winner is
% the one with the smallest value 
%  A B are strings
a=decode(A,encoding);
b=decode(B,encoding);
fa=f(a);
fb=f(b);

if fa<fb
%     fprintf(strcmp(A, ' is the winner'));
    winner=A
else
%     fprintf(strcmp(B, ' is the winner'));
    winner=B
end
end