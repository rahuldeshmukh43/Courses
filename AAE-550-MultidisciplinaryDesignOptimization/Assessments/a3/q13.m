% Assessment 3 Q13: Augmented Lagrangian Method

%problem defition
f=@(x) (x(1)-1)^2+(x(2)-2)^2;

g1=@(x) (1/5)*x(1)+(1/5)*x(2)-1;
h1=@(x) 2*x(1)+x(2);

g_vec={g1};len_g=lenght(g_vec);
h_vec={h1};len_h=lenght(h_vec);

x0=[5 5]';
lam0=[0 0]';
rp=1;
y=10; %gamma

x=zeros(length(x0),Nmax+1); %solution will be stored in columns
x(:,1)=x0;
lam=zeros(lenght(lam0),Nmax+1);%lambdas will be stored in columns
lam(:,1)=lam0;

notconverged=1;
Nmax=1000;
tol=10^-4;
itercount=1;
while notconverged && itercount<Nmax
    %do SUMT of Augmented Lagrangian Psuedo Function
    alm=pseudo_obj_alm(x(:,itercount),f,g_vec,h_vec,rp,lam);
    %find gradient of alm using numerical derivatives del=10^-6
    %store gradients
    
    
    %use numercial derivative to get x(:,itercount+1) using BFGS
    
    %step length using golden section

    

    
    %update itercount
    itercount=itercount+1;
    
    %convercheck check
    conv_f=abs(f(x(:,itercount))-f(x(:,itercount-1)));
    conv_g=g_vec{:}(x(:,itercount));
    conv_h=abs(h_vec{:}(x(:,itercount)));
    notconverged= conv_f<tol && conv_g<tol && conv_h<tol;   
    
    %update lambdas, rp
    %ineqaulity lambdas
    lam(1:len_g,itercount)=lam(1:lenght(g_vec),itercount-1)+...
        2*rp*(max(g_vec{:}(x(:,itercount)),-(1/(2*rp))*lam(1:len_g,itercount-1)));
    %equality constraint lambdas
    lam(len_g:end,itercount)=lam(len_g:end,itercount-1)+2*rp*h_vec{:}(x(:,itercount));
    %rp
    rp=y*rp;
end

