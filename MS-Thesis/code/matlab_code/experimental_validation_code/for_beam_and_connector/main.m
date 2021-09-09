% This Mainfile will 
%(0)print readme file with parameters for the current iteration
%(1)print coordinate file,
%(2)print aabaqus.py file,
%(3)run the python script using abaqus engine,
%(4)read output files generated by abaqus
%(5)generate plots for side-forces and axial forces wrt displacement
%(6)save all the output to a mat file for the whole parametric space

clc; clear all; close all;
%select the parent directory and read profile data
parent_dir = uigetdir('/export/home/a/deshmuk5/abaqus/MWspring/beam_connector_experimental/3065007/');

% define parameters which will vary and their possible values 
wire_dia = read_wire_dia_solid(parent_dir);% in mm
plate_thickness=10.0;%in mm

%friction_spring_stiff_top=[10^-4 10^-2 10^0 10^2 10^4];%in N/mm
friction_spring_stiff_top=[10^1];%in N/mm
%friction_spring_stiff_bottom=[10^-4 10^-2 10^0 10^2 10^4];%in N/mm
friction_spring_stiff_bottom=[10^1];%in N/mm
%springyoungmodulus=[160E3 170E3 175E3 180E3 190E3];%Mpa
springyoungmodulus=[250E3];%MPa
radialspring_stiffness=[10^4];%N/mm
radialspring_tension=[10^2];%N/mm


N=500; %number of points for the coordinate file

[X,Y,Z,ID,spring_initial_radius,spring_max_radius]=read_profile(parent_dir,N,wire_dia);% in mm
spring_height=max(Z);% in mm

%create output structure for storing the results
result = struct('wire_dia',{},'spring_height',{},'spring_max_radius',{},...%'pitchvar',{},'centervar',{},
    'sideforce',{},'axialforce',{},'sideforcemag',{},'displacement',{},...
    'friction_spring_stiff_top',{},'friction_spring_stiff_bottom',{},...%'degvar',{},
    'sideforcemag_exp',{},'axialforce_exp',{},'displacement_exp',{},...
    'springyoungmodulus',{},'radialspring_stiffness',{},'radialspring_tension',{});
matfilename = strcat(parent_dir,'/result.mat');
keys = {};
values = [];
save(matfilename,'result','keys','values','wire_dia','spring_height','spring_max_radius',...%'pitch_var','center_var',
    'friction_spring_stiff_top','friction_spring_stiff_bottom',...%'degvar',...
    'springyoungmodulus','radialspring_stiffness','radialspring_tension');
itotal = 1;
fprintf(strcat(parent_dir,'\n'));
for iE = 1:length(springyoungmodulus)
    for ikrt = 1:length(radialspring_tension)
        for ikrc=1:length(radialspring_stiffness)
            %             for idv =1:length(degvar)
            for ikft=1:length(friction_spring_stiff_top)
                for ikfb = 1:length(friction_spring_stiff_bottom)
                    %                         for iwd = 1:length(wire_dia)
                    %                             for ish = 1:length(spring_height)
                    %                                 for ismr = 1:length(spring_max_radius)
                    %                                     for ipv = 1:length(pitch_var)
                    %                                         for icv = 1:length(center_var)
                    %print cmd window to text file
                    diary(strcat(parent_dir,'/cmdstatus.txt'));
                    fprintf(strcat(num2str(itotal),'\n'));
                    %if itotal>xx
                    load(matfilename);
                    % to resume from a bad iteration  where x
                    %is the last successful iteration
                    fprintf(strcat('*** Starting ',char(32),num2str(itotal),' iteration at ',datestr(now),'***\n'));
                    %(0)create folder for this case and print out a readme file giving
                    %the values of the parameters for this case
                    foldername= strcat('E',num2str(iE),'krt',num2str(ikrt),'krc',num2str(ikrc),...%'dv',num2str(idv),
                        'kft',num2str(ikft),'kfb',num2str(ikfb));%'wd',num2str(iwd),'sh',num2str(ish),'smr',num2str(ismr),'pv',num2str(ipv),'cv',num2str(icv));
                    folderpath= strcat(parent_dir,'/runs/',foldername);
                    mkdir(folderpath);
                    filepath = strcat(folderpath,'/');
                    printreadme(filepath,wire_dia,spring_height,spring_max_radius,...%pitch_var(ipv),center_var(icv),
                    itotal,friction_spring_stiff_top(ikft),friction_spring_stiff_bottom(ikfb),...%degvar(idv),...
                        springyoungmodulus(iE),radialspring_stiffness(ikrc),radialspring_tension(ikrt));
                    fprintf('* Readme file printed *\n');
                    %(1)function call for printing the coordinate file
                    coordinatefile_bc(filepath,X,Y,Z,ID);
                    fprintf('* Coordinate file printed *\n');
                    %(2)function call for printing the abaqus.py file
                    printpyfile(filepath,N,wire_dia,spring_height,spring_max_radius,...
                        spring_initial_radius,plate_thickness,friction_spring_stiff_top(ikft),friction_spring_stiff_bottom(ikfb),...
                        springyoungmodulus(iE),radialspring_stiffness(ikrc),radialspring_tension(ikrt));
                    fprintf('* Abaqus python script printed *\n');
                    %(3)run the abaqus.py file
                    fprintf('* Submitting job to Abaqus *\n');
                    cmdstatus = system(strcat('abaqus cae noGUI=',filepath,'MyModel.py'));
                    while cmdstatus
                        %if license not fetched wait
                        fprintf(2,'!!Error: Abaqus Licence not fetched, trying again in 5 sec!!\n');
                        pause(5);% 5sec
                        cmdstatus = system(strcat('abaqus cae noGUI=',filepath,'MyModel.py')); %execute again
                    end
                    
                    %(4)function call for reading data from output
                    fprintf('* Reading results *\n');
                    [sideforcemag,sideforce,axialforce,displacement,sideforcemag_exp,axialforce_exp,displacement_exp]=readdata_solid(filepath,...
                        parent_dir,spring_height,wire_dia);
                    
                    %5(5)function call for generating plots
                    fprintf('* Generating plots *\n');
                    generateplots_solid(filepath,sideforcemag,sideforce,axialforce,displacement,...
                        sideforcemag_exp,axialforce_exp,displacement_exp);
                    
                    %(6)save results to the structure
                    result(itotal).wire_dia = wire_dia;
                    result(itotal).spring_height = spring_height;
                    result(itotal).spring_max_radius = spring_max_radius;
                    
%                     result(itotal).pitchvar =pitch_var(ipv);
%                     result(itotal).centervar = center_var(icv);
                    
                    result(itotal).sideforce=sideforce;
                    result(itotal).axialforce = axialforce';
                    result(itotal).displacement = displacement';
                    result(itotal).sideforcemag =sideforcemag;
                    
                    result(itotal).axialforce_exp = axialforce_exp;
                    result(itotal).displacement_exp = displacement_exp;
                    result(itotal).sideforcemag_exp =sideforcemag_exp;
                    
                    result(itotal).friction_spring_stiff_top = friction_spring_stiff_top(ikft);
                    result(itotal).friction_spring_stiff_bottom = friction_spring_stiff_bottom(ikfb);
%                     result(itotal).degvar = degvar(idv);
                    result(itotal).springyoungmodulus = springyoungmodulus(iE);
                    result(itotal).radialspring_stiffness = radialspring_stiffness(ikrc);
                    result(itotal).radialspring_tension = radialspring_tension(ikrt);
                    
                    values = 1:1:(itotal);
                    keys{itotal} = strcat(num2str(iE),num2str(ikrt),num2str(ikrc),num2str(ikft),num2str(ikfb));
                    save(matfilename,'result','values','keys','wire_dia','spring_height','spring_max_radius',...%'pitch_var','center_var',
                        'friction_spring_stiff_top','friction_spring_stiff_bottom',...%'degvar',...
                        'springyoungmodulus','radialspring_stiffness','radialspring_tension');
                    fprintf('* Saving results *\n');
                    fprintf(strcat('*** Completed',char(32),num2str(itotal),' iteration at ',datestr(now),' ***\n'));
                    fprintf('\n');
                    %end
                    %to resume from bad iteration
                    itotal = itotal+1;
                    diary off;
                    %                                         end
                    %                                     end
                    %                                 end
                    %                             end
                    %                         end
                end
            end
            %             end
        end
    end
end
diary off;
%create dictionary of keys and values
dict = containers.Map(keys,values);
%save dict to mat file which we can later use to access data
list = {'springyoungmodulus','radialspring_tension','radialspring_stiffness',...'degvar',
    'friction_spring_stiff_top','friction_spring_stiff_bottom'};%,'wire dia','spring height','spring max radius','pitch var','center var'};
save(matfilename,'result','dict','keys','values','wire_dia','spring_height','spring_max_radius',...%'pitch_var','center_var',
    'friction_spring_stiff_top','friction_spring_stiff_bottom',...%'degvar',...
    'springyoungmodulus','radialspring_stiffness','radialspring_tension','list');

setpref('Internet','SMTP_Server','smtp.purdue.edu');
setpref('Internet','E_mail','deshmuk5@purdue.edu');
sendmail('deshmuk5@purdue.edu','MATLAB mail',strcat('My Lord I have finished simulation of ',parent_dir,' at ',datestr(now)));

%exit;
