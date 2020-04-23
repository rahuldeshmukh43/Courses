function [C1,C2]=crossover(P1,P2,p,Pc)
% do crossover btw P1 and P2 using random numbers p
C1=[];
C2=[];

for i=1:length(P1)
    ip=p(i);
    if ip<=Pc
        C1=[C1,P2(i)];
        C2=[C2,P1(i)];        
    else
        C1=[C1,P1(i)];
        C2=[C2,P2(i)];        
    end
end

end