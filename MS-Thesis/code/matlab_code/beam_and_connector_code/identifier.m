function id = identifier(theta,t)
%this function returns an identifier corresponding to the theta position
%of the point

if t<=theta(2)
    id = 1;% only radial connectors
elseif t>theta(2) && t <= theta(1)+2*pi
    id = 2;%both radial and top and bottom connectors
elseif t>theta(end)-2*pi && t<theta(end-1)
    id = 4;%both radial and top and bottom connectors
elseif t>=theta(end-1)
    id = 5;% only radial connectors
else
    id =3;%only top and bottom connectors
end

end
