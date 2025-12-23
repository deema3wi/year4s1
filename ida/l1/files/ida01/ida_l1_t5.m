ida_l1_t4

% TASK 5 =================================================
plotYvsYhat_simple(y_tr, yhat_tr1, 'Лінійна (Train)');
plotYvsYhat_simple(y_te, yhat_te1, 'Лінійна (Test)');

plotYvsYhat_simple(y_tr, yhat_tr2, 'Квадратична (Train)');
plotYvsYhat_simple(y_te, yhat_te2, 'Квадратична (Test)');

function plotYvsYhat_simple(y_actual, y_pred, ttl)
    figure;
    scatter(y_actual, y_pred, 16, 'filled');
    grid on; hold on;

    mn = min([y_actual; y_pred]);
    mx = max([y_actual; y_pred]);
    plot([mn mx], [mn mx], '--', 'LineWidth', 1, 'Color', 'w');

    hold off;

    err  = y_actual - y_pred;
    rmse = sqrt(mean(err.^2));
    sse  = sum(err.^2);
    sst  = sum((y_actual - mean(y_actual)).^2);
    r2   = 1 - sse/sst;

    xlabel('Фактичні', 'Interpreter','none');
    ylabel('Прогнозовані', 'Interpreter','none');
    title(sprintf('%s | RMSE=%.4f | R^2=%.4f', ttl, rmse, r2), 'Interpreter','none');
end
