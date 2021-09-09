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

    indp=listdlg('PromptString','Select max three independent Variable','ListString',list);%,'SelectionMode','single');%list is always defined in the result file
    legendstr ={};
    for i =1:size(indp,2)
        indpvar{i}=eval(list{indp(i)});
        if isfloat(indpvar{i}(1))
            for n=1:length(indpvar{i})
                legendstr{i}(n) = {num2str(indpvar{i}(n))};
            end            
        elseif ischar(indpvar{i}(1))
            legendstr{i}=eval(list{i});
        end
    end
    fixedlist = list;
    fixedlist(indp) = '';
    fixed = zeros(length(list),1);
    ifixed = linspace(1,length(list),length(list));
    ifixed(indp) = []; 
    %prompt for user to select other variables
    
    for isel=1:length(fixedlist)
        displaylist={};
        temp=eval(fixedlist{isel});
        if isfloat(temp(1))            
            for n=1:length(temp)
                displaylist(n) = {num2str(temp(n))};
            end
        elseif ischar(temp(1))
            displaylist=temp;
        end
        tempsel = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');        
        fixed(ifixed(isel))=tempsel;
    end
    
    switch length(indp)
        case 1
            h=figure('units','normalized','outerposition',[0 0 1 1]);
            hold on;
            %suptitle(strcat({'Plots for varying'},{' '},list(indp)));
            p1=subplot(1,2,1);
            p2=subplot(1,2,2);
            for iplot=1:length(indpvar{1})
                fixed(indp)=iplot;
                keyname = '';
                for j =1:length(fixed)
                    keyname = strcat(keyname,num2str(fixed(j)));
                end
                ig = dict(keyname);
                plot(p1,result(ig).displacement,result(ig).sideforcemag);hold(p1,'on');
                title(p1,'Side-Forces vs Displacement');xlabel(p1,'Displacement(mm)');ylabel(p1,'Magnitude of Side-Force(N)');
                plot(p2,result(ig).displacement,result(ig).axialforce);hold(p2,'on');
                title(p2,'Axial-Force vs Displacement');xlabel(p2,'Displacement(mm)');ylabel(p2,'Axial-Force(N)');
            end
%             plot(p1,result(1).displacement_exp,result(1).sideforcemag_exp,':r');
%             plot(p2,result(1).displacement_exp,result(1).axialforce_exp,':r');
            count =1;pltleg={};P=[];
            for l=1:length(indpvar{1})                
                pltleg{count}=strcat(strrep(list{indp(1)},'_',' '),'=',legendstr{1}{l});
                count=count+1;
            end
%             pltleg{count} = 'MW testing';
            legend(p2,pltleg,'location','northwest');            
        case 2
            
            lst=["-" "--" "-."];
            mark =["o" "d" "^" "x" "*" "+"];
            %colr = ["blue" "magenta" "green" "yellow" "cyan" "black"];
            h=figure('units','normalized','outerposition',[0 0 1 1]);
            hold on;                    
            p1=subplot(1,2,1);
            p2=subplot(1,2,2);
            
            h2=figure('units','normalized','outerposition',[0 0 1 1]);
            hold on;
            p3=subplot(1,2,1);
            p4=subplot(1,2,2);

            for m=1:length(indpvar{2})
                for l=1:length(indpvar{1})
                    fixed(indp)=[l,m];
                    keyname = '';
                    for j =1:length(fixed)
                        keyname = strcat(keyname,num2str(fixed(j)));
                    end
                    ig = dict(keyname);
                    
                    plot(p1,result(ig).displacement,result(ig).sideforcemag,'Linestyle',lst{l},'Color','b');hold(p1,'on');
                    scatter(p1,result(ig).displacement(1:20:end),result(ig).sideforcemag(1:20:end),mark{m},'b');hold(p1,'on');
                    title(p1,'Side-Forces vs Displacement');
                    xlabel(p1,'Displacement(mm)');ylabel(p1,'Magnitude of Side-Force(N)');
                    
                    plot(p2,result(ig).displacement,result(ig).axialforce,'Linestyle',lst{l},'Color','b');hold(p2,'on');
                    scatter(p2,result(ig).displacement(1:20:end),result(ig).axialforce(1:20:end),mark{m},'b');hold(p1,'on');
                    title(p2,'Axial-Force vs Displacement');
                    xlabel(p2,'Displacement(mm)');ylabel(p2,'Axial-Force(N)');
                    
                    dia_pts = [result(ig).dia_start,result(ig).dia_max1,result(ig).dia_min1,result(ig).dia_max2,result(ig).dia_min2,result(ig).dia_max3,...
                        result(ig).dia_min3,result(ig).dia_max4,result(ig).dia_min4,result(ig).dia_max5,result(ig).dia_min5,result(ig).dia_max6,result(ig).dia_end];
                    splineangle = linspace(result(ig).angle_pts(1),result(ig).angle_pts(end),500);
                    dia = interp1(result(ig).angle_pts,dia_pts,splineangle,'cubic');
                    Z = interp1(result(ig).angle_pts,result(ig).height_pts,splineangle,'cubic');
                    
                    plot(p3,splineangle,Z,'Linestyle',lst{l});hold(p3,'on');
                    scatter(p3,result(ig).angle_pts,result(ig).height_pts,mark{m});hold(p3,'on');
                    title(p3,'Angle vs Height');
                    xlabel(p3,'Angle (deg)');ylabel(p3,'Height(mm)');
                    
                    plot(p4,splineangle,dia,'Linestyle',lst{l});hold(p4,'on');
                    scatter(p4,result(ig).angle_pts,dia_pts,mark{m});hold(p4,'on');
                    title(p4,'Angle vs Dia');
                    xlabel(p4,'Angle (deg)');ylabel(p4,'Diameter(mm)');
                end
            end           
            
%             plot(p1,result(1).displacement_exp,result(1).sideforcemag_exp,':r');
%             plot(p2,result(1).displacement_exp,result(1).axialforce_exp,':r');
            count =1;pltleg={};P=[];
            for l=1:length(indpvar{1})
                P(count)=plot(p2,nan,nan,lst{l},'Color','k');
                pltleg{count}=strcat(strrep(list{indp(1)},'_',' '),'=',legendstr{1}{l});
                count=count+1;
            end
            for m=1:length(indpvar{2})
                P(count)=plot(p2,nan,nan,mark{m},'Color','k');
                pltleg{count}=strcat(strrep(list{indp(2)},'_',' '),'=',legendstr{2}{m});
                count=count+1;
            end
            legend(P,pltleg,'location','northwest');
        case 3
            %make subplot for the sideforce vs disp and axialforce vs disp wrt indepent variable
            h=figure('units','normalized','outerposition',[0 0 1 1]);
            lst=["-" "--" "-."];
            mark =["o" "d" "^" "x" "*" "+"];
            colr = ["blue" "magenta" "green" "yellow" "cyan" "black"];
            hold on;                        
            p1=subplot(1,2,1);
            p2=subplot(1,2,2);
            h2=figure('units','normalized','outerposition',[0 0 1 1]);
            hold on;
            p3=subplot(1,2,1);
            p4=subplot(1,2,2);
            for n=1:length(indpvar{3})
                for m=1:length(indpvar{2})
                    for l=1:length(indpvar{1})
                        fixed(indp)=[l,m,n];
                        keyname = '';
                        for j =1:length(fixed)
                            keyname = strcat(keyname,num2str(fixed(j)));
                        end
                        ig = dict(keyname);
                        
                        plot(p1,result(ig).displacement,result(ig).sideforcemag,'Linestyle',lst{l},'Color',colr{n});hold(p1,'on');
                        scatter(p1,result(ig).displacement(1:20:end),result(ig).sideforcemag(1:20:end),mark{m},colr{n});hold(p1,'on');
                        title(p1,'Side-Forces vs Displacement');
                        xlabel(p1,'Displacement(mm)');ylabel(p1,'Magnitude of Side-Force(N)');
                        
                        plot(p2,result(ig).displacement,result(ig).axialforce,'Linestyle',lst{l},'Color',colr{n});hold(p2,'on');
                        scatter(p2,result(ig).displacement(1:20:end),result(ig).axialforce(1:20:end),mark{m},colr{n});hold(p1,'on');
                        title(p2,'Axial-Force vs Displacement');
                        xlabel(p2,'Displacement(mm)');ylabel(p2,'Axial-Force(N)');
                        
                        dia_pts = [result(ig).dia_start,result(ig).dia_max1,result(ig).dia_min1,result(ig).dia_max2,result(ig).dia_min2,result(ig).dia_max3,...
                            result(ig).dia_min3,result(ig).dia_max4,result(ig).dia_min4,result(ig).dia_max5,result(ig).dia_min5,result(ig).dia_max6,result(ig).dia_end];
                        splineangle = linspace(result(ig).angle_pts(1),result(ig).angle_pts(end),500);
                        dia = interp1(result(ig).angle_pts,dia_pts,splineangle,'cubic');
                        Z = interp1(result(ig).angle_pts,result(ig).height_pts,splineangle,'cubic');
                        
                        plot(p3,splineangle,Z,'Linestyle',lst{l},'Color',colr{n});hold(p3,'on');
                        scatter(p3,result(ig).angle_pts,result(ig).height_pts,mark{m},colr{n});hold(p3,'on');
                        title(p3,'Angle vs Height');
                        xlabel(p3,'Angle (deg)');ylabel(p3,'Height(mm)');
                        
                        plot(p4,splineangle,dia,'Linestyle',lst{l},'Color',colr{n});hold(p4,'on');
                        scatter(p4,result(ig).angle_pts,dia_pts,mark{m},colr{n});hold(p4,'on');
                        title(p4,'Angle vs Dia');
                        xlabel(p4,'Angle (deg)');ylabel(p4,'Diameter(mm)');
                        
                        
                    end
                end
            end
            
%             plot(p1,result(1).displacement_exp,result(1).sideforcemag_exp,':r');
%             plot(p2,result(1).displacement_exp,result(1).axialforce_exp,':r');
            count =1;pltleg={};P =[];
            for l=1:length(indpvar{1})
                P(count)=plot(p2,nan,nan,lst{l},'Color','k');
                pltleg{count}=strcat(strrep(list{indp(1)},'_',' '),'=',legendstr{1}{l});
                count=count+1;
            end
            for m=1:length(indpvar{2})
                P(count)=plot(p2,nan,nan,mark{m},'Color','k');
                pltleg{count}=strcat(strrep(list{indp(2)},'_',' '),'=',legendstr{2}{m});
                count=count+1;
            end
            for n=1:length(indpvar{3})
                P(count)=plot(p2,nan,nan,colr{n});
                pltleg{count}=strcat(strrep(list{indp(3)},'_',' '),'=',legendstr{3}{n});
                count=count+1;
            end
            legend(P,pltleg,'location','northwest');            
    end
    
    %ask to save
    prompt= {'Save Plot? Y-1/N-0', 'Filename'};
    selkey ='';
    for i=1:length(fixed)
       if sum(ismember(indp,i))>0     %i==indp(1) || i==indp(2)
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
    if do
        close(h);
    end
end
