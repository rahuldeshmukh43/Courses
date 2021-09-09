function printreadme(filepath,wire_dia,...%pitch_var,center_var,
    itotal,friction_spring_stiff_top,friction_spring_stiff_bottom,...%degvar,...
    springyoungmodulus,radialspring_stiffness,radialspring_tension,...
    dia_start,dia_max1,dia_max2,dia_max3,dia_max4,dia_max5,dia_max6,dia_end,...
    dia_min1,dia_min2,dia_min3,dia_min4,dia_min5,angle_pts,height_pts)

% This function prints out a readme file with information about the
% geometric parameters for this case
fID = fopen(strcat(filepath,'readme.txt'),'w');

fprintf(fID,'For this Iteration we have the following paramteres\n');
fprintf(fID,strcat(' springyoungmodulus:  ',num2str(springyoungmodulus),'Mpa\n'));
fprintf(fID,strcat('radialspring_stiffness:  ',num2str(radialspring_stiffness),'\n'));
fprintf(fID,strcat('radialspring_tension:  ',num2str(radialspring_tension),'\n'));
% fprintf(fID,strcat('degvar:  ',num2str(degvar),'degrees \n'));
fprintf(fID,strcat('K friction spring Top:  ',num2str(friction_spring_stiff_top),'\n'));
fprintf(fID,strcat('K friction spring Bottom:  ',num2str(friction_spring_stiff_bottom),'\n'));
fprintf(fID,strcat('Wire diameter:  ',num2str(wire_dia),'\n'));

% fprintf(fID,strcat('Spring Max Radius: ',num2str(spring_max_radius),'\n'));
% fprintf(fID,strcat('Spring height: ',num2str(spring_height),'\n'));
% fprintf(fID,strcat('Pitch Variation parameter:  ',num2str(pitch_var),'\n'));
% fprintf(fID,strcat('Center Variation parameter:   ',center_var,'\n'));

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