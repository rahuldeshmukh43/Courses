%code for finding peaks and valleys which will be used to construct cubic interpolation
clc; clear all; close all;
inchestomm =25.4;
filename= uigetfile('*.xlsx');
 

angle= xlsread(filename,'A:A');
angle = (angle/(360.0))*2*pi;
angle  = angle -(angle(1))*ones(length(angle),1);% to make th starting point at theta=0 because the abaqus code uses that convention, this will not change the geometry of the spring in any sense

diameter = xlsread(filename,'B:B');
height = xlsread(filename,'C:C');

% figure(1);
% findpeaks(diameter,angle,'MinPeakProminence',0.01,'Annotate','extents');
[dia_max,dia_max_loc]=findpeaks(diameter,angle,'MinPeakProminence',0.01,'Annotate','extents');

% figure(2);
% findpeaks(-1*diameter,angle,'MinPeakProminence',0.01,'Annotate','extents');
[dia_min,dia_min_loc]=findpeaks(-1*diameter,angle,'MinPeakProminence',0.01,'Annotate','extents');
dia_min = -1*dia_min;% as we had multiplied by -1 earlier to find the min peaks

figure(3);
hold on;
plot(angle,diameter,'b-');xlabel('angle(rad)');ylabel('Diameter(in)');
scatter(dia_max_loc,dia_max,'ro');
scatter(dia_min_loc,dia_min,'r+');

splineangle = linspace(angle(1),angle(end),500);
splinedia = spline([angle(1);dia_max_loc; dia_min_loc;angle(end)],...
    [diameter(1);dia_max; dia_min;diameter(end)],splineangle);
% plot(splineangle,splinedia,'m');

Linterpdia = interp1([angle(1);dia_max_loc; dia_min_loc;angle(end)],...
    [diameter(1);dia_max; dia_min;diameter(end)],splineangle,'linear');
% plot(splineangle,Linterpdia,'c');

Cinterpdia = interp1([angle(1);dia_max_loc; dia_min_loc;angle(end)],...
    [diameter(1);dia_max; dia_min;diameter(end)],splineangle,'cubic');
plot(splineangle,Cinterpdia,'k');
legend('actual','max','min','Cubic interp');

figure(4);
findpeaks(height,angle,'MinPeakProminence',100000,'Annotate','extents');

hold on;
plot(angle,height,'-b');
for i=1:length(dia_max_loc)
    scatter(dia_max_loc(i),height(find(angle==dia_max_loc(i))),'ro');    
end
for i=1:length(dia_min_loc)
    scatter(dia_min_loc(i),height(find(angle==dia_min_loc(i))),'r+');    
end

temph =[];tempa=sort([angle(1);dia_max_loc; dia_min_loc;angle(end)]);
for i=1:length(tempa)
    temph = [temph; height(find(angle==tempa(i)))];
end
Cinterph = interp1(tempa,...
    temph,splineangle,'cubic');

plot(splineangle,Cinterph,'k');xlabel('angle(rad)');ylabel('Height(in)');
hold off;
fprintf('dia_max\n');
display(dia_max*inchestomm);
fprintf('dia_min\n');display(dia_min*inchestomm);
% fprintf('sorted_dia\n');display(sort([dia_max_loc;dia_min_loc])*inchestomm);
fprintf('angle_pts\n');display(sort(tempa)*180/pi);
fprintf('height_pts\n');display((temph)*inchestomm);
