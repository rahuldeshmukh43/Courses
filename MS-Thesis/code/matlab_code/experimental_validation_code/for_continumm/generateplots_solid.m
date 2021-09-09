function sideforcemag = generateplots_solid(filepath,sideforcemag,sideforce,axialforce,displacement,...
    sideforcemag_exp,axialforce_exp,displacement_exp)
% This function will generate the force vs displacement curves for side
% forces and axial forces and save these plots as png to the filepath

mkdir (strcat(filepath,'plots'));
filepath = strcat(filepath,'plots/');

figure(1)
plot(displacement,axialforce,'b');
hold on;
plot(displacement_exp,axialforce_exp,':r');
title('Axial Force vs Displacement');
xlabel('Displacement (mm)');
ylabel('Axial Force (N)');
legend('computed','experimental','Location','northwest');
saveas(gcf,strcat(filepath,'axialforce_vs_displacement.png'));
hold off;

figure(2)
plot(displacement,sideforce(:,1));
title('Side Force (Fx) vs Displacement');
xlabel('Displacement (mm)');
ylabel('Side Force Fx (N)');
saveas(gcf,strcat(filepath,'sideforcex_vs_displacement.png'));

figure(3)
plot(displacement,sideforce(:,3));
title('Side Force(Fz) vs Displacement');
xlabel('Displacement (mm)');
ylabel('Side Force Fz (N)');
saveas(gcf,strcat(filepath,'sideforcez_vs_displacement.png'));

figure(4)
% sideforcemag = sqrt(sideforce(:,1).^2+sideforce(:,3).^2);
plot(displacement,sideforcemag);
hold on;
plot(displacement_exp,sideforcemag_exp,':r');
title('Side Force (Magnitude) vs Displacement');
xlabel('Displacement (mm)');
ylabel('Side Force magnitude (N)');
legend('computed','experimental','Location','northwest');
saveas(gcf,strcat(filepath,'sideforcemag_vs_displacement.png'));
hold off;

close all;
end