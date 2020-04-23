function f = GAfunc(x)
% Objective function for AAE550 : HW 3  Problem 4
% Truss Optimization: Minimize Weight
% Design variables: x(1:3): Material: type =discrete
%                   x(4:6): cross sectional area in mm^2: type= continuous
% Created By Rahul Deshmukh
% email : deshmuk5@purdue.edu
% PUID: 0030004932


A = 1E-6*x(4:6); % converted cross sectional areas to [m^2]


% set Geometric entities
% Lengths and Angles of Bars
L(1) = (5*cosd(75)-3.3*sind(75))/(cosd(124));	% length of bar 1 [m]
theta2 = atand(L(1)*sind(49)/(L(1)*cosd(49)-3))+180;
L(2) = L(1)*sind(49)/sind(theta2);               % length of bar 2 [m]
L(3) = 3.3/cosd(75)-L(1)*sind(49)/cosd(75);   % length of bar 3 [m]


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

% Total Mass
mass=sum(rho.*A.*L);

% account for constraints using penalty approach
sigma = stressHW3(A,E); % row vector
g = abs(sigma)./sigma_y-1;

% cj= [1 1000 100];
cj= [1 100000 100];
% using Exterior Quadratic
% phi= sum(cj.*(max(0,g).^2));

% using Exterior Linear
% phi= sum(cj.*max(0,g));

% using Exterior step-Linear
Pj=zeros(length(g),1);
for j=1:length(g)
    if g(j)<=0
        Pj(j)=0;
    else
        Pj(j)=cj(j)*(1+g(j));
    end
end
phi= sum(Pj);


rp=1000;
f= mass +rp*phi;

end

