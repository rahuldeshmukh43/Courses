function [g,geq]=nonLcon_f2(x,e)

    g= ((exp(x(2))+6)/e)-1;
    geq=[];
end