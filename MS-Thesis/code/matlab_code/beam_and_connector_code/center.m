function [xc,yc] = center(center_var,spring_height,Z,degvar)
%this function will give the position of the center point taking into
%center_var=[ 'none', 'sin', 'cos' 'linear'] 
% spring_height is the total height of the spring
% Z is the current height of the point
% output needs to be x and y corrdinates of the center

bottom_center = [0.0,0.0,0.0];
%degvar = 4;%in degrees
switch center_var
    case 'none'
        xc = 0.0;
        yc = 0.0;
    case 'halfsin'
        %assuming half-sin behavior and both top and bottom centers are at
        %0,0,0 and 0,0,h also center variation is only reflected in x direction
        yc =0.0;
        fun= @(z) (spring_height/2)*(tand(degvar))*sin(pi*z/spring_height);
        xc = fun(Z);
    case 'fullsin'
        %assuming half-sin behavior and both top and bottom centers are at
        %0,0,0 and 0,0,h also center variation is only reflected in x direction
        yc =0.0;
        fun= @(z) (spring_height/4)*(tand(degvar))*sin(2*pi*z/spring_height);
        xc = fun(Z);
    case 'halfcos'
        %code later, here end-centers will also change in xy plane
        %fun= @(z) (spring_height/2)*(tand(degvar))*cos(pi*z/spring_height);%half cosine variation
    case 'fullcos'
        %code later, here end-centers will also change in xy plane
        %fun= @(z) (spring_height/4)*(tand(degvar))*cos(pi*z/spring_height);%half cosine variation
    case 'linear'
        % assuming the bottom center remains at (0,0,0) position but both
        % of the pug centers will be in the same position on z axis
        % regardless of the spring center variation so basically this will
        % not lead to any changes in the py file
        
        %the slope of the axis will be degvar from vertical center
        %variation assumed only in x direction
        xc=0.0+Z*tand(degvar);
        yc=0.0;
        
        %for the case when mid height is on axis 
%         xc= (Z-spring_height/2)*tand(degvar);
%         yc=0.0;
    case 'quadratic'
        %assuming one root at z=0 and another root at 2/3H and at z=H
        %variation is H*tand(degvar)
        xc=(3*tand(degvar)/spring_height)*Z*(Z-(2/3)*spring_height);
        yc=0.0;
end

end