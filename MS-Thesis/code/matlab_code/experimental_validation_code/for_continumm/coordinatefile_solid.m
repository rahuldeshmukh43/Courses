function coordinatefile_solid(filepath,X,Y,Z)
id =1;%irrelevant in the solid code
fID = fopen(strcat(filepath,'coordinatefile.txt'),'w');
for i=1:length(X)
    fprintf(fID,'%d %d %f %f %f\n',id,i,X(i),Y(i),Z(i));  
end
fclose(fID);
end