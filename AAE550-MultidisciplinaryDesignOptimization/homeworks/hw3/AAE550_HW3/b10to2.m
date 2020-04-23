function gen=b10to2(b10,bits)
% Function for generating gray code for an integer values array b10
% with bits as bit length per var
gen=[];
for i=1:size(b10,1)
    igen=[];
    for j=1:size(b10,2)
       bin=de2bi(b10(i,j),bits(j));
       gray=bi2gray(bin);
       igen=[igen,gray];
    end
    gen=[gen;igen];
end
end


function gray= bi2gray(bin)
% function for converting a binary array to gray code
bits=length(bin);
gray=[bin(1)];
for i=2:length(bin)
    gray=[gray,(xor(bin(i-1),bin(i)))];    
end
end