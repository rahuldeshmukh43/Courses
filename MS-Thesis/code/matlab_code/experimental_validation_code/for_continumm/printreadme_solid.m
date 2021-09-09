function printreadme(filepath,wire_dia,spring_height,spring_max_radius,itotal,...
    springyoungmodulus,friction_coeff,springmeshSize,platemeshSize) 
% This function prints out a readme file with information about the
% geometric parameters for this case
fID = fopen(strcat(filepath,'readme.txt'),'w');
fprintf(fID,'For this Iteration we have the following paramteres\n');
fprintf(fID,strcat('Spring Youngs Modulus:  ',num2str(springyoungmodulus),'Mpa\n'));
fprintf(fID,strcat('Coefficient of Friction :  ',num2str(friction_coeff),'\n'));
fprintf(fID,strcat('Spring Mesh Size :  ',num2str(springmeshSize),'\n'));
fprintf(fID,strcat('Plate Mesh Size :  ',num2str(platemeshSize),'\n'));
fprintf(fID,strcat('Wire Diameter:  ',num2str(wire_dia),'\n'));
fprintf(fID,strcat('Spring Height:  ',num2str(spring_height),'\n'));
fprintf(fID,strcat('Spring Max Radius: ',num2str(spring_max_radius),'\n'));
fprintf(fID,strcat('Global Iteration number: ',num2str(itotal),'\n'));
fclose(fID);
end