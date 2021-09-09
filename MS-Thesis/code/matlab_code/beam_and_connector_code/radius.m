function r = radius(spring_initial_radius,spring_max_radius,theta,t,start)
%this function will give the radius (from center) for the curretn piont ie
%at a particular theta taking into account the variation in radius

%radius increases linearly wrt t(i)
%{
%feature to add start angle for dead turn ie non constant radius in dead
%turn, but we will have to change the limits for the quantities in the
%following if-else branches
a=pi;%angle till which we have a constant radius in the dead turn (should be less than or equal to pi)
th1=a;
th2=theta(7)-a;
%}
%start of increase in radius %cannot be more than theta(2)
%start = theta(2);%cannot be more than theta2 %%for run1
%start = theta(2)-pi/2;%cannot be more than theta2 %%for run2
thetamid = theta(ceil(length(theta)/2));
slope = (spring_max_radius-spring_initial_radius)/(thetamid-start);
if t<=start
    r = spring_initial_radius;
elseif t>start && t<=thetamid
    r  = spring_initial_radius+slope*(t-start);
% elseif t>theta(2) && t <= thetamid
%     r  = spring_initial_radius+slope*(t-start);
elseif t>thetamid && t< thetamid+(thetamid-start)
    r = spring_max_radius-slope*(t-thetamid);   
elseif t>=thetamid+(thetamid-start)
    r = spring_initial_radius;    
end

end