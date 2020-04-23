function sigma = stressHW3(A,E)
% This function assembles the stiffness matrix and computes the stress in
% each element of the three-bar truss in AAE 550, HW 3, part II, Fall 2017.
%
% The function returns a three-element vector "sigma"; each element is the computed
% stress in each truss element.  The input is the three-element vector "A";
% each element is the cross-sectional area of each truss element.  Values
% for Young's modulus are input as parameters

% fixed values
P = 120000;                     % applied load [N]

% Lengths and Angles of Bars
L(1) = (5*cosd(75)-3.3*sind(75))/(cosd(124));	% length of bar 1 [m]
theta2 = atand(L(1)*sind(49)/(L(1)*cosd(49)-3))+180;

L(2) = L(1)*sind(49)/sind(theta2);               % length of bar 2 [m]
L(3) = 3.3/cosd(75)-L(1)*sind(49)/cosd(75);   % length of bar 3 [m]

theta1 = 49;
theta3 = 75;

% local stiffness matrices
K1 = [cosd(-theta1) sind(-theta1)]'*(E(1)*A(1)/L(1))*[cosd(-theta1) sind(-theta1)];
K2 = [cosd(-theta2) sind(-theta2)]'*(E(2)*A(2)/L(2))*[cosd(-theta2) sind(-theta2)];
K3 = [cosd(theta3+90) sind(theta3+90)]'*(E(3)*A(3)/L(3))*[cosd(theta3+90) sind(theta3+90)];

% global (total) stiffness matrix:
K = K1 + K2 + K3;

% load vector (note lower case to distinguish from P)
theta4 = -110;
p = P*[cosd(theta4) sind(theta4)]';

% compute displacements (u(1) = x-displacement on figure; u(2) =
% y-displacement on figure)
u = K \ p;

% change in element length under load
DL(1) = sqrt((-L(1)*cosd(-theta1)-u(1))^2 + (-L(1)*sind(-theta1)-u(2))^2) - L(1);
DL(2) = sqrt((-L(2)*cosd(-theta2)-u(1))^2 + (-L(2)*sind(-theta2)-u(2))^2) - L(2);
DL(3) = sqrt((-L(3)*cosd(theta3+90)-u(1))^2 + (-L(3)*sind(theta3+90)-u(2))^2) - L(3);

% stress in each element
sigma = E .* DL ./ L;
