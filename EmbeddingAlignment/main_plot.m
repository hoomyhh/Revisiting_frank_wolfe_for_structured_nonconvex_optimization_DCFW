
% Comparing the Performance of DCFW, FW-M, FW-K, theoretical bound using
% FW_Gap and #SVD Calculations

close all

load results

hfig = figure('Name','EmbeddingAlignment','NumberTitle','off','Position',[100,100,1000,285]);

subplot(131)

loglog(infoFWK.gap);
hold on;
loglog(infoFWM.gap);
loglog(infoDCFW.outer.tk, infoDCFW.outer.gap);

xlabel('iterations','Interpreter','latex','fontsize',14)
ylabel('Frank-Wolfe Gap','Interpreter','latex','fontsize',14)
hleg = legend('FW-K', 'FW-M', 'DC-FW', 'DC-CGS');
hleg.Interpreter = 'latex';
hleg.FontSize = 14;
hleg.Location = 'SouthWest';
axis tight;
ax = gca
ax.XTick = 10.^(0:4);


subplot(132)

semilogy(infoFWK.counter_subgradg + infoFWK.counter_lmo, infoFWK.gap);
hold on;
loglog(infoFWM.counter_subgradg + infoFWM.counter_lmo + infoFWM.counter_obj, infoFWM.gap);
loglog(infoDCFW.outer.counter_subgradg + infoDCFW.outer.counter_lmo, infoDCFW.outer.gap)  ;

xlabel('SVD calculations','Interpreter','latex','fontsize',14)
xlim([0,4e4])
axis tight;
ax = gca;
ax.XTick = (0:4)*1e4;
xlim([0,4e4])


subplot(133)

semilogy(infoFWK.time, infoFWK.gap);
hold on;
loglog(infoFWM.time, infoFWM.gap);
loglog(infoDCFW.outer.time, infoDCFW.outer.gap)  ;
axis tight;

xlabel('time (sec)','Interpreter','latex','fontsize',14)
ax = gca;
ax.XTick = (0:300:1500);
xlim([0,1300])

for t = 1:3

    subplot(1,3,t)
    ax = gca;
    set(findall(ax, 'Type', 'line'),'LineWidth',3);
    ax.FontSize = 14;
    ax.TickLabelInterpreter = 'latex';
    ax.TickDir = 'out';
    grid on; grid minor; grid minor;
    set(gca,'TickDir','out')
    set(gca,'LineWidth',0.75,'TickLength',[0.02 0.02]);
    %     ax.XTick = 10.^(-10:10);
    %     ax.YTick = 10.^(-100:2:100);
    ax.Box = 'on';

end

