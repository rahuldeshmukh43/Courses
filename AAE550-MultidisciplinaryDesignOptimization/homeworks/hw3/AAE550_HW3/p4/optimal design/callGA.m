% this file provides input variables to the genetic algorithm
% upper and lower bounds, and number of bits chosen for "egg-crate" problem
% Modified on 11/10/09 by Bill Crossley.
clc;
close all;
clear all;

options = goptions([]);

%lb ub for area in [mm^2]
lb= 100;
ub= 1000;

vlb = [1 1 1 lb*ones(1,3) 30 30];	%Lower bound of each gene - all variables
vub = [4 4 4 ub*ones(1,3) 60 85];	%Upper bound of each gene - all variables
bits =[2 2 2 10 10 10 10 10];	%number of bits describing each gene - all variables


[x,fbest,stats,nfit,fgen,lgen,lfit]= GA550('GAfunc',[ ],options,vlb,vub,bits);

format short;
x
A = 1E-6*x(4:6); % converted cross sectional areas to [m^2]

% set Geometric entities
% Lengths and Angles of Bars
L(1) = (5*cosd(x(end))-3.3*sind(x(end)))/(cosd(sum(x(end-1:end))));	% length of bar 1 [m]
theta2 = atand(L(1)*sind(x(end-1))/(L(1)*cosd(x(end-1))-3))+180;
L(2) = L(1)*sind(x(end-1))/sind(theta2);               % length of bar 2 [m]
L(3) = 3.3/cosd(x(end))-L(1)*sind(x(end-1))/cosd(x(end));   % length of bar 3 [m]

theta1 = x(end-1);
theta3 = x(end);

% set Material properties based on x(1:3)
rho=[];
E=[];
sigma_y=[];
for i=1:3    
    if x(i) == 1
        % Aluminum
        E = [E, 68.9E9]; % [Pa]
        rho = [rho,2700]; % [km/m^3]
        sigma_y = [sigma_y,55.2E6]; % [Pa]
    elseif x(i) == 2
        % Titanium
        E = [E,116E9];    % [Pa]
        rho = [rho,4500];    % [kg/m^3]
        sigma_y = [sigma_y,140E6];    % [Pa]
    elseif x(i) == 3
        % Steel
        E = [E,205E9];    % [Pa]
        rho = [rho,7872];    % [kg/m^3]
        sigma_y = [sigma_y,285E6];    % [Pa]
    elseif x(i)==4
        % Nickel
        E = [E,207E9];    % [Pa]
        rho = [rho,8800];    % [kg/m^3]
        sigma_y = [sigma_y,59E6];    % [Pa]
    end    
end

% resolution
resolution=(vub-vlb)./(2.^bits-1);

%chromosome length
lchrom=sum(bits);
Npop=4*lchrom;
Pm=(lchrom+1)/(2*Npop*lchrom);

% Total Mass
mass=sum(rho.*A.*L) % in KG
 
% account for constraints using penalty approach
sigma = stressHW3(A,E,x(end-1:end)) %  [PA]
g = abs(sigma)./sigma_y-1
g<=0

figure(2);
hold on;
plot([0, -L(3)*sind(theta3)],[-3.3,-3.3+L(3)*cosd(theta3)]) %truss-3
plot([-5+3, -5+3-L(2)*cosd(180-theta2)],[0, -L(2)*sind(180-theta2) ]) %truss-2
plot([-5, -5+3-L(2)*cosd(180-theta2)],[0, -L(2)*sind(180-theta2) ]) %truss-1
