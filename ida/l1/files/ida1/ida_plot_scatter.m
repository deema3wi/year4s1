function ida_plot_scatter(x, y, xlbl, ylbl, ttl)
figure;
scatter(x, y, 'filled');
grid on;
xlabel(xlbl, 'Interpreter','none');
ylabel(ylbl, 'Interpreter','none');
title(ttl, 'Interpreter','none');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
end
