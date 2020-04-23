% Created by Rahul Deshmukh
% PUID: 0030004932
% email: deshmuk5@purdue.edu

function gen=b10to2(b10,bits)
% Function for generating gray code from an integer valued matrix b10 with
% bits as bit length per var
% Input: b10: Npop x 2 matrix of integer elements
%        bits: array of length Nvar with information of bit 
%              length per design variable
% Output: gen: Npop x sum(bits) matrix of gray coded 
%              generation,format: not string but numbers 0 and 1s 
gen=[];
for i=1:size(b10,1)
    igen=[];
    for j=1:size(b10,2)
       bin=de2bi(b10(i,j),bits(j)); % first convert to binary
       gray=bi2gray(bin); % convert to gray-code
       igen=[igen,gray];
    end
    gen=[gen;igen];
end
end


function gray= bi2gray(bin)
% function for converting a binary array to gray code
% Input: bin: a binary array of 1 and 0 of length bits(i)
% Output: gray: gray coded equivalent of the binary word, 
%               format:array of 1 and 0
bits=length(bin); 
gray=[bin(1)]; 
for i=2:length(bin)
    gray=[gray,(xor(bin(i-1),bin(i)))];    
end
end