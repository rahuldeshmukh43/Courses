function coordinatefile_bc(filepath,X,Y,Z,ID)

fID = fopen(strcat(filepath,'coordinatefile.txt'),'w');
for i=1:length(X)
    fprintf(fID,'%d %d %f %f %f\n',ID(i),i,X(i),Y(i),Z(i));  
end
fclose(fID);
end
