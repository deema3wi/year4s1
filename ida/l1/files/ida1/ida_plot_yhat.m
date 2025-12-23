function ida_plot_yhat(y_actual, y_pred, ttl)
figure;
scatter(y_actual, y_pred, 16, 'filled');
grid on; hold on;

mn = min([y_actual(:); y_pred(:)]);
mx = max([y_actual(:); y_pred(:)]);
plot([mn mx], [mn mx], '--', 'LineWidth', 1, 'Color', 'w');

hold off;

[rmse, r2] = ida_metrics(y_actual, y_pred);

xlabel('Фактичні', 'Interpreter','none');
ylabel('Прогнозовані', 'Interpreter','none');
title(sprintf('%s | RMSE=%.4f | R2=%.4f', ttl, rmse, r2), 'Interpreter','none');
end
