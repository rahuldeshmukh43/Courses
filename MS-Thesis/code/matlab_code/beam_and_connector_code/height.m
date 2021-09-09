function H = height(pitch_var,spring_height,t,theta)
%this function will give the height of the point at a particular theta
%taking into account the variation in pitch 
%pitch varies linearly and therefore height varies quadratically wrt t(i)

m=spring_height/((theta(ceil(length(theta)/2)))-(theta(2)))^2; %rate of change of pitch
a = (pitch_var/100)*m*((theta(ceil(length(theta)/2))+theta(2))/2-theta(2));%quantity to be added to displace the quarterpoints

tmid =theta(ceil(length(theta)/2));
tqt = (tmid+theta(2))/2;
t3qt = (tmid+theta(end-1))/2;
m1 = ((m*(tqt-theta(2))+a)/(tqt-theta(2)));
m2 = ((m*(tmid-theta(2))-(m*(tqt-theta(2))+a))/(tmid-tqt));
m3 = ((m*(tmid-theta(2))-(m*(tqt-theta(2))-a))/(tmid-t3qt));
m4 = (((m*(tqt-theta(2))-a)-0)/(t3qt-theta(end-1)));
f1 = @(x) m1*(x-theta(2));
f2 = @(x) m2*(x-tmid) + m*(tmid-theta(2));
f3 = @(x) m3*(x-tmid) + m*(tmid-theta(2));
f4 = @(x) m4*(x-theta(end-1));

%to plot the whole pitch function
% f5 = @(x) m*(x-theta(2));
% f6 = @(x) (-m*(tmid-theta(2))/(theta(end-1)-tmid))*(x-theta(end-1));
% d = 2;
% x1 = theta(2):1/d:tqt;
% x2= tqt:1/d:tmid;
% x3=tmid:1/d:t3qt;
% x4=t3qt:1/d:theta(end-1);
% figure(1)
% hold on 
% plot(x1,f1(x1));plot(x1,f5(x1));
% plot(x2,f2(x2));plot(x2,f5(x2));
% plot(x3,f3(x3));plot(x3,f6(x3));
% plot(x4,f4(x4));plot(x4,f6(x4));
% hold off;

if t<=theta(2)
    H = 0;
elseif t>theta(2) && t<=tqt
    H = integral(f1,theta(2),t);
elseif t>tqt && t<=tmid
    H = integral(f1,theta(2),tqt)+integral(f2,tqt,t);
elseif t>tmid && t<=t3qt
    H = integral(f1,theta(2),tqt)+integral(f2,tqt,tmid)+integral(f3,tmid,t);
elseif t>t3qt && t<=theta(end-1)
    H = integral(f1,theta(2),tqt)+integral(f2,tqt,tmid)+integral(f3,tmid,t3qt)+integral(f4,t3qt,t);
elseif t>theta(end-1)
    H = spring_height;
end

end