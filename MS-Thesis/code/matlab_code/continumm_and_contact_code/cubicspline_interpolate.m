function [X,Y,Z,spring_height,spring_max_radius,spring_initial_radius]=cubicspline_interpolate(filepath,N,dia_start,dia_max1,dia_max2,dia_max3,dia_max4,dia_max5,dia_max6,dia_end,...
    dia_min1,dia_min2,dia_min3,dia_min4,dia_min5,angle_pts,height_pts,wire_dia)

dia_pts = [dia_start,dia_max1,dia_min1,dia_max2,dia_min2,dia_max3,dia_min3,dia_max4,dia_min4,dia_max5,dia_min5,dia_max6,dia_end];

splineangle = linspace(angle_pts(1),angle_pts(end),N);
theta = splineangle*pi/180;% in radians
dia = interp1(angle_pts,dia_pts,splineangle,'cubic'); 
Z = interp1(angle_pts,height_pts,splineangle,'cubic');
spring_height=max(Z);

radius  = (dia-wire_dia*ones(1,length(dia)))/2.0;
spring_max_radius=max(radius);spring_initial_radius=min(radius);
X = (radius.*cos(theta));
Y = (radius.*sin(theta));

f1=figure(1);
plot(splineangle,dia,'b');hold on;
scatter(angle_pts,dia_pts,'ro');
xlabel('angle(deg)');ylabel('diameter(mm)');title('Angle vs Dia(outer)');hold off;
saveas(f1,strcat(filepath,'/angle_vs_dia.png'));

f2=figure(2);
plot(splineangle,Z,'b');hold on;
scatter(angle_pts,height_pts,'ro');
xlabel('angle(deg)');ylabel('height(mm)');title('Angle vs Height(central)');hold off;
saveas(f2,strcat(filepath,'/angle_vs_height.png'));

f3=figure(3);
plot3(X,Y,Z,'o:','MarkerSize',wire_dia);
view(3);xlabel('X(mm)');ylabel('Y(mm)');zlabel('Z(mm)');
saveas(f3,strcat(filepath,'/geometry.png'));
close all;                                                        
end