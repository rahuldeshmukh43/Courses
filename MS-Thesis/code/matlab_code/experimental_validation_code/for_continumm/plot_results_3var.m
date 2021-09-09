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
exp_present = inputdlg({'experimental data present?'},'Experiment data present dlg box',1,{'0'});
while do
    
    indp=listdlg('PromptString','Select max three independent Variable','ListString',list);%,'SelectionMode','single');%list is always defined in the result file
    legendstr ={};
    
    for i =1:size(indp,2)
%{
%         switch list{indp(i)}
%             case 'degvar'
%                 indpvar{i}=degvar;
%                 for n=1:length(degvar)
%                     legendstr{i}(n) = {num2str(degvar(n))};
%                 end
%             case 'friction spring stiff top'
%                 indpvar{i}=friction_spring_stiff_top;
%                 for n=1:length(friction_spring_stiff_top)
%                     legendstr{i}(n) = {num2str(friction_spring_stiff_top(n))};
%                 end
%             case 'friction spring stiff bottom'
%                 indpvar{i}=friction_spring_stiff_bottom;
%                 for n=1:length(friction_spring_stiff_bottom)
%                     legendstr{i}(n) = {num2str(friction_spring_stiff_bottom(n))};
%                 end
%             case 'wire dia'
%                 indpvar{i}=wire_dia;
%                 for n=1:length(wire_dia)
%                     legendstr{i}(n) = {num2str(wire_dia(n))};
%                 end
%             case 'spring height'
%                 for n=1:length(spring_height)
%                     legendstr{i}(n) = {num2str(spring_height(n))};
%                 end
%                 indpvar{i}=spring_height;
%             case 'spring max radius'
%                 for n=1:length(spring_max_radius)
%                     legendstr{i}(n) = {num2str(spring_max_radius(n))};
%                 end
%                 indpvar{i}=spring_max_radius;
%             case 'pitch var'
%                 for n=1:length(pitch_var)
%                     legendstr{i}(n) = {num2str(pitch_var(n))};
%                 end
%                 indpvar{i}=pitch_var;
%             case 'center var'
%                 legendstr{i} = center_var;
%                 indpvar{i}=center_var;
%         end
%}
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
        %{
%         switch fixedlist{isel}
%             case 'degvar'
%                 for n=1:length(degvar)
%                     displaylist(n) = {num2str(degvar(n))};
%                 end
%                 temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
%                 fixed(1)=temp;
%             case 'friction spring stiff top'
%                 for n=1:length(friction_spring_stiff_top)
%                     displaylist(n) = {num2str(friction_spring_stiff_top(n))};
%                 end
%                 temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
%                 fixed(2)=temp;
%             case 'friction spring stiff bottom'
%                 for n=1:length(friction_spring_stiff_bottom)
%                     displaylist(n) = {num2str(friction_spring_stiff_bottom(n))};
%                 end
%                 temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
%                 fixed(3)=temp; 
%             case 'wire dia'
%                 for n=1:length(wire_dia)
%                     displaylist(n) = {num2str(wire_dia(n))};
%                 end
%                 temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
%                 fixed(4)=temp;                
%             case 'spring height'
%                 for n=1:length(spring_height)
%                     displaylist(n) = {num2str(spring_height(n))};
%                 end
%                 temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
%                 fixed(5)=temp;
%             case 'spring max radius'
%                 for n=1:length(spring_max_radius)
%                     displaylist(n) = {num2str(spring_max_radius(n))};
%                 end
%                 temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
%                 fixed(6)=temp;
%             case 'pitch var'
%                 for n=1:length(pitch_var)
%                     displaylist(n) = {num2str(pitch_var(n))};
%                 end
%                 temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
%                 fixed(7)=temp;
%             case 'center var'
%                 displaylist = center_var;%already a cell array
%                 temp = listdlg('PromptString',strcat({'Select'},{' '},fixedlist(isel)),'ListString',displaylist,'SelectionMode','single');
%                 fixed(8)=temp;
%         end
%}
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
            colr = ["blue" "magenta" "green" "yellow" "cyan" "black"];
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
                plot(p1,result(ig).displacement,result(ig).sideforcemag,'Linestyle','-','Color',colr{iplot});hold(p1,'on');
                title(p1,'Side-Forces vs Displacement');xlabel(p1,'Displacement(mm)');ylabel(p1,'Magnitude of Side-Force(N)');
                plot(p2,result(ig).displacement,result(ig).axialforce,'Linestyle','-','Color',colr{iplot});hold(p2,'on');
                title(p2,'Axial-Force vs Displacement');xlabel(p2,'Displacement(mm)');ylabel(p2,'Axial-Force(N)');
            end
            if exp_present{1} == '1'
            plot(p1,result(1).displacement_exp,result(1).sideforcemag_exp,':r');
            plot(p2,result(1).displacement_exp,result(1).axialforce_exp,':r');
            end
            count =1;pltleg={};P=[];
            for l=1:length(indpvar{1})
                P(count)=plot(p2,nan,nan,'Color',colr{l});
                pltleg{count}=strcat(strrep(list{indp(1)},'_',' '),'=',legendstr{1}{l});
                count=count+1;
            end
            if exp_present{1} == '1'
                P(count)=plot(p2,nan,nan,':r');
                pltleg{count}=strcat('Experimental data');
                count=count+1;
            end
%             pltleg{count} = 'MW testing';
            legend(p2,pltleg,'location','northwest');            
        case 2
            h=figure('units','normalized','outerposition',[0 0 1 1]);
            lst=["-" "--" "-."];
            mark =["o" "d" "^" "x" "*" "+"];
            %colr = ["blue" "magenta" "green" "yellow" "cyan" "black"];
            hold on;                    
            p1=subplot(1,2,1);
            p2=subplot(1,2,2);
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
                end
            end           
            if exp_present{1} =='1'
            plot(p1,result(1).displacement_exp,result(1).sideforcemag_exp,':r');
            plot(p2,result(1).displacement_exp,result(1).axialforce_exp,':r');
            end
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
            if exp_present{1} == '1'
                P(count)=plot(p2,nan,nan,':r');
                pltleg{count}=strcat('Experimental data');
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
                    end
                end
            end
            if exp_present{1}=='1'
            plot(p1,result(1).displacement_exp,result(1).sideforcemag_exp,':r');
            plot(p2,result(1).displacement_exp,result(1).axialforce_exp,':r');
            end
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
            if exp_present{1} == '1'
                P(count)=plot(p2,nan,nan,':r');
                pltleg{count}=strcat('Experimental data');
                count=count+1;
            end
            legend(P,pltleg,'location','northwest');            
    end
    
    %ask about plotiing from another mat file
        
    
    prompt = {'Plot from other MAT files?'};
    definput = {'0'};
    morefileans = inputdlg(prompt,'More MAT file dlg box',1,definput);
    if morefileans{1} == '1'
        [filename2,foldername2] = uigetfile('*.mat','Select the Mat file');
        fullfilename2= fullfile(foldername2,filename2);
        file2 = load(fullfilename2);
        
        plot(p1,file2.result.displacement,file2.result.sideforcemag,'+k','Linestyle','-');
        P(count)=plot(p2,file2.result.displacement,file2.result.axialforce,'+k','Linestyle','-');
        pltleg{count}='beam connector model';
        count = count+1;
        legend(P,pltleg,'location','northwest')
    end
%     exp_present = inputdlg({'experimental data present?'},'Experiment data present dlg box',1,{'0'});
    
    set(findall(p1, 'Type', 'Line'),'LineWidth',4);
    set(p1,'FontSize',20);
    set(findall(p2, 'Type', 'Line'),'LineWidth',4);
    set(p2,'FontSize',20);
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
        save_dir = uigetdir('/export/home/a/deshmuk5/abaqus/MWspring/');
        saveas(gcf,strcat(save_dir,'/',savefileans{2},'.png'));
    end
    do = inputdlg('Plot More? Y-1/N-0','Plot More dlg box',1,{'1'});
    do = str2num(do{1});
    if do
        close(h);       
    end
end
