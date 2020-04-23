%HW1 problem (1)
%Rahul Deshmukh PUID: 0030004932
%deshmuk5@purdue.edu
%---------------begin----------------------%
%gradient of enrgy function
function [PE,gradPE]=hw1_p1_PEwtgrad(u)
%input: u is a col vector
%output: PE: scalar value of potential energy
%        gradPE: gradient of potential energy function

%definition of constants: E,A,L,phi,P
E=17.3*10^6;%psi
d1=0.65;%in
A1=pi*(d1/2)^2;%sq in
A2=A1;
d3=0.8;%in
A3=pi*(d3/2)^2;%sq in
fttoin=12;% conversion factor 
L1=3*fttoin;
L2=L1;
L3=(5-(2.75/2))*fttoin;
phi=90-(180/pi)*acos((2.75/2)/3);
P=14000;%lbs

theta=32;%degrees
p=P*[sind(theta);cosd(theta)];

K1=[cosd(-1*phi);sind(-1*phi)]*(E*A1/L1)*[cosd(-1*phi),sind(-1*phi)];
K2=[cosd(phi);sind(phi)]*(E*A2/L2)*[cosd(phi),sind(phi)];
K3=[0;sind(90)]*(E*A3/L3)*[0,sind(90)];

K=K1+K2+K3;

PE=(1/2)*u'*K*u-p'*u;
gradPE=(1/2)*(K+K')*u-p;

end