% TASK 7 =================================================
vars = T.Properties.VariableNames;

y = T.V_9; 
y = y(:);

isNum = varfun(@isnumeric, T, 'OutputFormat','uniform');
predNames = vars(isNum);
predNames = setdiff(predNames, {'V_9','V-9'}, 'stable');

Xall = T{:, predNames};
Xall = double(Xall);

Xtr_all = Xall(trainIdx, :);
Xte_all = Xall(testIdx,  :);

ytr_all = y(trainIdx); ytr_all = ytr_all(:);
yte_all = y(testIdx);  yte_all = yte_all(:);

maskXtr = builtin('all', isfinite(Xtr_all), 2);
maskXte = builtin('all', isfinite(Xte_all), 2);

mtr = isfinite(ytr_all) & maskXtr;
mte = isfinite(yte_all) & maskXte;

Xtr_all = Xtr_all(mtr, :);
ytr_all = ytr_all(mtr);

Xte_all = Xte_all(mte, :);
yte_all = yte_all(mte);

s = std(Xtr_all, 0, 1);
keep = isfinite(s) & (s > 0);

Xtr_all = Xtr_all(:, keep);
Xte_all = Xte_all(:, keep);
predKept = predNames(keep);

Xtr = [ones(size(Xtr_all,1),1), Xtr_all];
Xte = [ones(size(Xte_all,1),1), Xte_all];

a = lsqminnorm(Xtr, ytr_all);

yhat_tr = Xtr * a;
yhat_te = Xte * a;

[rmse_tr, r2_tr] = metrics(ytr_all, yhat_tr);
[rmse_te, r2_te] = metrics(yte_all, yhat_te);

disp(table(size(Xtr,2), rmse_tr, r2_tr, rmse_te, r2_te, ...
    'VariableNames', {'p','RMSE_Train','R2_Train','RMSE_Test','R2_Test'}));

plotYvsYhat_simple(ytr_all, yhat_tr, 'Linear(all numeric) (Train)');
plotYvsYhat_simple(yte_all, yhat_te, 'Linear(all numeric) (Test)');

function [rmse, r2] = metrics(y, yhat)
    err = y - yhat;
    rmse = sqrt(mean(err.^2));
    sse = sum(err.^2);
    sst = sum((y - mean(y)).^2);
    r2 = 1 - sse/sst;
end

function plotYvsYhat_simple(y_actual, y_pred, ttl)
    figure;
    scatter(y_actual, y_pred, 16, 'filled');
    grid on; hold on;

    mn = min([y_actual; y_pred]);
    mx = max([y_actual; y_pred]);
    plot([mn mx], [mn mx], '--', 'LineWidth', 1, 'Color', 'w');

    hold off;

    [rmse, r2] = metrics(y_actual, y_pred);

    xlabel('Фактичні', 'Interpreter','none');
    ylabel('Прогнозовані', 'Interpreter','none');
    title(sprintf('%s | RMSE=%.4f | R2=%.4f', ttl, rmse, r2), 'Interpreter','none');
end