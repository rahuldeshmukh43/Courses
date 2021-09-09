function [X,Y,Z,ID,spring_initial_radius,spring_max_radius]=read_profile(parent_dir,N,wire_dia)
%This function will read the Profile data from profile.xlsx file and will
%output N number of x y z coordinates. 
inchestomm =25.4;
filename = strcat(parent_dir,'/profile.xlsx');

% coils_accum = xlsread(filename,'A:A');
angle= xlsread(filename,'A:A');
angle = (angle/(360.0))*2*pi;
angle  = angle -(angle(1))*ones(length(angle),1);% to make th starting point at theta=0 because the abaqus code uses that convention, this will not change the geometry of the spring in any sense

diameter = xlsread(filename,'B:B');
height = xlsread(filename,'C:C');
% pitch = xlsread(filename,'C:C');
% feed = xlsread(filename,'D:D');
% coils = xlsread(filename,'E:E');

% height =[];angle=[];
% for i=1:length(pitch)
%    height = [height sum(pitch(1:i))]; 
%    angle= [angle coils_accum(i)*2*pi];
% end

theta = linspace(angle(1),angle(end),N);
splinedia = spline(angle,diameter,theta); %outer dia in inches
radius =(splinedia-(wire_dia/(inchestomm))*(ones(1,length(splinedia))))/2.0;
%radius =(splinedia+(1.4-(wire_dia/(inchestomm)))*(ones(1,length(splinedia))))/2.0;
%if profiler is giving inner dia information
%radius=(splinedia+(wire_dia/(inchestomm))*(ones(1,length(splinedia))))/2.0;
%if profiler is giving centerline information.
%radius =(splinedia)/2.0;

spring_initial_radius = inchestomm*min(radius);%is used to decide the size of the puck
spring_max_radius = inchestomm*max(radius);%is used to decide the size of the plate

Z = inchestomm*spline(angle,height,theta);
X = inchestomm*(radius.*cos(theta));
Y = inchestomm*(radius.*sin(theta));

%ID
ID = zeros(length(theta),1);
for i=1:length(theta)
   if theta(i)<=pi
       ID(i) = 1;
   elseif theta(i)<=2*pi && theta(i)>pi
       ID(i) = 2;
   elseif theta(i)>=theta(end)-2*pi && theta(i)<theta(end)-pi
       ID(i) = 4;
   elseif theta(i)>=theta(end)-pi
       ID(i) = 5;
   else 
       ID(i) = 3;
   end
end
% 3D plot of the spring
f1=figure(1);
plot3(X,Y,Z,'o:','MarkerSize',wire_dia);
view(3);xlabel('X(mm)');ylabel('Y(mm)');zlabel('Z(mm)');
saveas(f1,strcat(parent_dir,'/geometry.png'));

%plots as obtained on profiler
f2=figure(2);
plot(height,diameter,'.',Z/inchestomm,splinedia);%not the centerline dia
xlabel('height(in)');ylabel('diameter(in)');
saveas(f2,strcat(parent_dir,'/height_vs_diameter.png'));

f3=figure(3);
plot(angle,diameter,'.',theta,splinedia);%not the centerline dia
xlabel('theta(rad)');ylabel('diameter(in)');
saveas(f3,strcat(parent_dir,'/theta_vs_diameter.png'));
close all;

end