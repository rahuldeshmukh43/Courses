%this script will read and produce plots for side force and axial force for
%the desired variables
%load file
clc; clear all;close all;

[filename,foldername] = uigetfile('*.mat','Select the Mat file');
fullfilename= fullfile(foldername,filename);
load(fullfilename);
% list = {'Wire Diameter','Spring Height','Spring Max Radius',...
%     'Pitch Variation parameter','Center Variation parameter'};
do=1;

while do
    indp=listdlg('PromptString','Select The independent Variable','ListString',list,'SelectionMode','single');%list is always defined in the result file
    legendstr ={};
    switch list{indp}
        case 'friction spring stiff top'
            indpvar=friction_spring_stiff_top;
            for n=1:length(friction_spring_stiff_top)
                legendstr(n) = {num2str(friction_spring_stiff_top(n))};
            end
        case 'friction spring stiff bottom'
            indpvar=friction_spring_stiff_bottom;
            for n=1:length(friction_spring_stiff_bottom)
                legendstr(n) = {num2str(friction_spring_stiff_bottom(n))};
            end
        case 'wire dia'
            indpvar=wire_dia;
            for n=1:length(wire_dia)
                legendstr(n) = {num2str(wire_dia(n))};
            end
        case 'spring height'
            for n=1:length(spring_height)
                legendstr(n) = {num2str(spring_height(n))};
            end
            indpvar=spring_height;
        case 'spring max radius'
            for n=1:length(spring_max_radius)
                legendstr(n) = {num2str(spring_max_radius(n))};
            end
            indpvar=spring_max_radius;
        case 'pitch var'
            for n=1:length(pitch_var)
                legendstr(n) = {num2str(pitch_var(n))};
            end
            indpvar=pitch_var;
        case 'center var'            
            legendstr = center_var;            
            indpvar=center_var;
    end
    
    fixedlist = list;
    fixedlist(indp) = '';
    fixed = zeros(length(list),1);
    %prompt for user to select other variables
    
    for isel=1:length(list)-1
        displaylist={};
        switch fixedlist{isel}
            case 'friction spring stiff top'
                for n=1:length(friction_spring_stiff_top)
                    displaylist(n) = {num2str(friction_spring_stiff_top(n))};
                end
                temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
                fixed(1)=temp;
            case 'friction spring stiff bottom'
                for n=1:length(friction_spring_stiff_bottom)
                    displaylist(n) = {num2str(friction_spring_stiff_bottom(n))};
                end
                temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
                fixed(2)=temp; 
            case 'wire dia'
                for n=1:length(wire_dia)
                    displaylist(n) = {num2str(wire_dia(n))};
                end
                temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
                fixed(3)=temp;
                
            case 'spring height'
                for n=1:length(spring_height)
                    displaylist(n) = {num2str(spring_height(n))};
                end
                temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
                fixed(4)=temp;
            case 'spring max radius'
                for n=1:length(spring_max_radius)
                    displaylist(n) = {num2str(spring_max_radius(n))};
                end
                temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
                fixed(5)=temp;
            case 'pitch var'
                for n=1:length(pitch_var)
                    displaylist(n) = {num2str(pitch_var(n))};
                end
                temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
                fixed(6)=temp;
            case 'center var'
                displaylist = center_var;%already a cell array
                temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
                fixed(7)=temp;
        end
    end
    
    %make subplot for the sideforce vs disp and axialforce vs disp wrt indepent variable
    h=figure('units','normalized','outerposition',[0 0 1 1]);
    hold on;
    suptitle(strcat({'Plots for varying'},{' '},list(indp)));
    p1=subplot(1,2,1);
    p2=subplot(1,2,2);
    for iplot=1:length(indpvar)
        fixed(indp)=iplot;
        keyname = '';
        for j =1:length(fixed)
            keyname = strcat(keyname,num2str(fixed(j)));
        end
        ig = dict(keyname);
        %     subplot(1,2,1);
        plot(p1,result(ig).displacement,result(ig).sideforcemag);hold(p1,'on');
        title(p1,'Side-Forces vs Displacement');xlabel(p1,'Displacement(mm)');ylabel(p1,'Magnitude of Side-Force(N)');
        %     subplot(1,2,2);
        plot(p2,result(ig).displacement,result(ig).axialforce);hold(p2,'on');
        title(p2,'Axial-Force vs Displacement');xlabel(p2,'Displacement(mm)');ylabel(p2,'Axial-Force(N)');
    end
    [f1,d1]=plot_mw_side();
    plot(p1,d1,f1,':r');
    [f2,d2]=plot_mw_axial();
    plot(p2,d2,f2,':r');
    legendstr(length(legendstr)+1) = {'MW testing'}; 
    legend(p2,legendstr,'Location','SouthEast');
    %ask to save
    prompt= {'Save Plot? Y-1/N-0', 'Filename'};
    selkey ='';
    for i=1:length(fixed)
       if i==indp
           selkey=strcat(selkey,'V');
       else
           selkey=strcat(selkey,num2str(fixed(i)));
       end
    end
    definput = {'0',strcat('Plot_for_',selkey)};
    [savefileans] = inputdlg(prompt,'Save Plot dlg Box',[2 20],definput);
    if savefileans{1}=='1'
        saveas(gcf,strcat(foldername,'plots/',savefileans{2},'.png'));
    end
    do = inputdlg('Plot More? Y-1/N-0','Plot More dlg box',1,{'1'});
    do = str2num(do{1});
    close(h);
end
