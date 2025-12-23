function ida_plot_split_scatter(x_tr, y_tr, x_te, y_te, xlbl, ylbl, ttl)
figure;
hold on;
scatter(x_tr, y_tr, 20, 'magenta', 'filled');
scatter(x_te, y_te, 20, 'cyan', 'filled');
hold off;
grid on;
xlabel(xlbl, 'Interpreter','none');
ylabel(ylbl, 'Interpreter','none');
title(ttl, 'Interpreter','none');
legend('Train','Test','Location','best');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
end
