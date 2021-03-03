%AAE:550 HW1 P2 
% Rahul Deshmukh
% PUID: 00 
%%
clc; clear all;
format long;
%----------------------begin------------------------------------%
%define symbols for design variables
syms cr b a; %root chord, wing span,angle of attack

%define related independent terms
V=53.6;% m/sec
rho=1.134;%kg/m^3

Tr=0.4; %taper ratio
ct=Tr*cr; % tip chord
C_la=2*pi;% lift curve slope
e=0.9; %efficiency
a_L0= -3;
S=((ct+cr)/2)*b;
AR=b^2/S;

a1=0.14;
k=(1-(1+ (pi*AR/C_la) )*a1)/((1+ (pi*AR/C_la) )*a1);

C_La=C_la/( (1+ (C_la/(pi*AR)) )*(1+k));

C_L=C_La*(pi/180)*(a-a_L0); %need to convert angles to radians here

q=0.5*rho*V^2;

C_Di=(C_L^2)/(pi*AR*e);

%find symbolic expressions for objective and constraints
%objective function min Di
Di=q*S*C_Di;
vpa(simplify(Di))  % 0.033949298754935466466210448881266*b^2*(a + 3.0)^2
% old  111.44897353114793675304812440719*b^2*(a + 3.0)^2

% constraint on L>=9500 N
L=q*S*C_L;
vpa(simplify(L)) % 12.50454558912777413506277959447*b^2*(a + 3.0)
% 716.45768698595102198388079976053*b^2*(a + 3.0)

%constraint on 0.7=<C_L<=0.9
vpa(simplify(C_L)) % (0.01096622711232150957648276777764*b*(a + 3.0))/cr
% (0.6283185307179586476925286766559*b*(a + 3.0))/cr

%%

% using variable scaling somehow
