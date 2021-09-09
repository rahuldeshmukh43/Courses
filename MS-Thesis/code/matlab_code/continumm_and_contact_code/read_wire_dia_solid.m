function [wire_dia]=read_wire_dia_solid(parent_dir)

fID = fopen(strcat(parent_dir,'/wiredia.txt'),'r');
wire_dia =fscanf(fID,'%f');% wire_dia in mm
fclose(fID);

end