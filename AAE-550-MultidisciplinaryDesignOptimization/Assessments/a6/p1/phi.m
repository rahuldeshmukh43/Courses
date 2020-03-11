function op=phi(x,a,fl_min,f1,f2)


iphi=zeros(2,1);
iphi(1)= ((f1(x)-fl_min(1))/fl_min(1))^2;
iphi(2)= ((f2(x)-fl_min(2))/fl_min(2))^2;
op=a'*iphi;
end