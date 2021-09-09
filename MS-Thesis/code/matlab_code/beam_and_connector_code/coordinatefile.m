function coordinatefile(filepath,N,theta,wire_dia,spring_height,spring_max_radius,pitch_var,center_var,spring_initial_radius,start,degvar)
% This funciton generates a text file with X Y Z coordinates of the spring
% points in the location filepath

t = linspace(theta(1),theta(length(theta)),N);%angle array
X  = zeros(N,1);
Y  = zeros(N,1);
Z  = zeros(N,1);
fID = fopen(strcat(filepath,'coordinatefile.txt'),'w');
for i=1:length(t)
    r = radius(spring_initial_radius,spring_max_radius,theta,t(i),start); 
    Z(i) = height(pitch_var,spring_height,t(i),theta);
    [xc,yc] = center(center_var,spring_height,Z(i),degvar);
    X(i) = xc + (r)*cos(t(i));
    Y(i) = yc + (r)*sin(t(i));
    id = identifier(theta,t(i));
    fprintf(fID,'%d %d %f %f %f\n',id,i,X(i),Y(i),Z(i));   
end
fclose(fID);
figure(1)
plot3(X,Y,Z,'-o');
%{
% plotstr = {strcat('geometry for wire dia:',num2str(wire_dia));...
%     strcat('height:',num2str(spring_height));...
%     strcat('max radius:',num2str(spring_max_radius));...
%     strcat('pitch var:',num2str(pitch_var));...
%     strcat('center var:',center_var)};
% %annotation('textbox','String',plotstr,'FitBoxToText','on');
% legend(plotstr,'location','EastOutside');
%}
saveas(gcf,strcat(filepath,'geometry.png'));
close all;
end
