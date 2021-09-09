function [f,d]=plot_mw_side()
%This function will output the force anad displacement vectors for side
%force behavior as extracted from MW data
f= [0 45 60 65 80 100 108 100 76 90]*4.45;%converting lbs to newtons
d = -[0 -0.42 (6.1-8.42) (5.4-8.42) (4.5-8.42) (3.6-8.42)...
    (3.1-8.42) (2.6-8.42) (1.75-8.42) (1.5-8.42)]*25.4;%converting to mm
end