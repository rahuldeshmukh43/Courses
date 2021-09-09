function printreadme_solid(filepath,wire_dia,itotal,...
    springyoungmodulus,friction_coeff,springmeshSize,platemeshSize,...
    dia_start,dia_max1,dia_max2,dia_max3,dia_max4,dia_max5,dia_max6,dia_end,...
    dia_min1,dia_min2,dia_min3,dia_min4,dia_min5,angle_pts,height_pts) 
% This function prints out a readme file with information about the
% geometric parameters for this case
fID = fopen(strcat(filepath,'readme.txt'),'w');

fprintf(fID,'For this Iteration we have the following paramteres\n');
fprintf(fID,strcat('Spring Youngs Modulus:  ',num2str(springyoungmodulus),'Mpa\n'));
fprintf(fID,strcat('Coefficient of Friction :  ',num2str(friction_coeff),'\n'));
fprintf(fID,strcat('Spring Mesh Size :  ',num2str(springmeshSize),'\n'));
fprintf(fID,strcat('Plate Mesh Size :  ',num2str(platemeshSize),'\n'));
fprintf(fID,strcat('Wire Diameter:  ',num2str(wire_dia),'\n'));

% fprintf(fID,strcat('Spring Height:  ',num2str(spring_height),'\n'));
% fprintf(fID,strcat('Spring Max Radius: ',num2str(spring_max_radius),'\n'));
dia_pts = [dia_start,dia_max1,dia_min1,dia_max2,dia_min2,dia_max3,dia_min3,dia_max4,dia_min4,dia_max5,dia_min5,dia_max6,dia_end];
fprintf(fID,'dia_pts=\n');
fprintf(fID,'[');fprintf(fID,'%6.3f\t',dia_pts);fprintf(fID,'] in mm\n');
fprintf(fID,'angle_pts=\n');
fprintf(fID,'[');fprintf(fID,'%6.3f\t',angle_pts);fprintf(fID,'] in deg \n');
fprintf(fID,'height_pts=\n');
fprintf(fID,'[');fprintf(fID,'%6.3f\t',height_pts);fprintf(fID,'] in mm\n');
fprintf(fID,strcat('Global Iteration number: ',num2str(itotal),'\n'));

fclose(fID);
end