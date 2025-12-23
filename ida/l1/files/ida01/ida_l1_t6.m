ida_l1_t5

%% TASK 6 =================================================

vars = T.Properties.VariableNames;

x1 = getcol(T, {'V_5','V-5'});
x2 = getcol(T, {'V_13','V-13'});
y = getcol(T, {'V_9','V-9'});

x1 = x1(:); x2 = x2(:); y = y(:);

isNum = varfun(@isnumeric, T, 'OutputFormat','uniform');
cand = vars(isNum);
cand = setdiff(cand, {'V_5','V_13','V_9','V-5','V-13','V-9'}, 'stable');

modelNames = ["M1 linear", ...
              "M2 quad", ...
              "M3 full quad", ...
              "M4 cubic", ...
              "M5 mixed cubic"];

rows = numel(cand) * 5;

VnName = strings(rows,1);
ModelId = zeros(rows,1);
ModelName = strings(rows,1);
p = zeros(rows,1);
RMSE_Train = zeros(rows,1);
R2_Train = zeros(rows,1);
RMSE_Test = zeros(rows,1);
R2_Test = zeros(rows,1);

r = 0;
bestRMSE = inf;
best_ttl = "";
best_ytr = [];
best_yhat_tr = [];
best_yte = [];
best_yhat_te = [];


for i = 1:numel(cand)
    x3 = T{:, cand{i}};
    x3 = x3(:);

    x1_tr = x1(trainIdx); x2_tr = x2(trainIdx); x3_tr = x3(trainIdx); y_tr = y(trainIdx);
    x1_te = x1(testIdx);  x2_te = x2(testIdx);  x3_te = x3(testIdx);  y_te = y(testIdx);

    x1_tr = x1_tr(:); x2_tr = x2_tr(:); x3_tr = x3_tr(:); y_tr = y_tr(:);
    x1_te = x1_te(:); x2_te = x2_te(:); x3_te = x3_te(:); y_te = y_te(:);

    mtr = isfinite(y_tr) & isfinite(x1_tr) & isfinite(x2_tr) & isfinite(x3_tr);
    mte = isfinite(y_te) & isfinite(x1_te) & isfinite(x2_te) & isfinite(x3_te);

    x1tr = x1_tr(mtr); x2tr = x2_tr(mtr); x3tr = x3_tr(mtr); ytr = y_tr(mtr);
    x1te = x1_te(mte); x2te = x2_te(mte); x3te = x3_te(mte); yte = y_te(mte);

    for m = 1:5
        Xtr = designMatrix3(m, x1tr, x2tr, x3tr);
        Xte = designMatrix3(m, x1te, x2te, x3te);

        a = lsqminnorm(Xtr, ytr);

        yhat_tr = Xtr * a;
        yhat_te = Xte * a;

        [rmse_tr, r2_tr] = metrics(ytr, yhat_tr);
        [rmse_te, r2_te] = metrics(yte, yhat_te);

        r = r + 1;

        VnName(r) = string(cand{i});
        ModelId(r) = m;
        ModelName(r) = modelNames(m);
        p(r) = size(Xtr,2);

        RMSE_Train(r) = rmse_tr;
        R2_Train(r) = r2_tr;
        RMSE_Test(r) = rmse_te;
        R2_Test(r) = r2_te;

        if rmse_te < bestRMSE
            bestRMSE = rmse_te;
            best_ttl = sprintf('%s, %s', string(cand{i}), modelNames(m));
            best_ytr = ytr;         best_yhat_tr = yhat_tr;
            best_yte = yte;         best_yhat_te = yhat_te;
        end

    end
end

t6 = table(VnName, ModelId, ModelName, p, RMSE_Train, R2_Train, RMSE_Test, R2_Test);
t6_sorted = sortrows(t6, 'RMSE_Test', 'ascend');

disp('TOP-10 за RMSE_Test:');
disp(t6_sorted(1:min(10,height(t6_sorted)), :));

plotYvsYhat_simple(best_ytr, best_yhat_tr, best_ttl + " (Train)");
plotYvsYhat_simple(best_yte, best_yhat_te, best_ttl + " (Test)");

function X = designMatrix3(m, x1, x2, x3)
    n = numel(x1);
    switch m
        case 1
            X = [ones(n,1), x1, x2, x3];
        case 2
            X = [ones(n,1), x1, x2, x3, x1.^2, x2.^2, x3.^2];
        case 3
            X = [ones(n,1), x1, x2, x3, x1.^2, x2.^2, x3.^2, x1.*x2, x1.*x3, x2.*x3];
        case 4
            X = [ones(n,1), x1, x2, x3, x1.^2, x2.^2, x3.^2, x1.^3, x2.^3, x3.^3];
        case 5
            X = [ones(n,1), x1, x2, x3, ...
                 x1.*x2, x1.*x3, x2.*x3, ...
                 (x1.^2).*x2, x1.*(x2.^2), ...
                 (x1.^2).*x3, x1.*(x3.^2), ...
                 (x2.^2).*x3, x2.*(x3.^2)];
        otherwise
            error('Unknown model id');
    end
end

function [rmse, r2] = metrics(y, yhat)
    err = y - yhat;
    rmse = sqrt(mean(err.^2));
    sse = sum(err.^2);
    sst = sum((y - mean(y)).^2);
    r2 = 1 - sse/sst;
end

function v = getcol(T, candidates)
    vn = T.Properties.VariableNames;
    k = find(ismember(vn, candidates), 1, 'first');
    if isempty(k)
        error('Column not found: %s', strjoin(candidates, ' or '));
    end
    v = T{:, vn{k}};
end

%% коре
corr(T.V_8, T.V_9, 'Rows','complete')

function plotYvsYhat_simple(y_actual, y_pred, ttl)
    figure;
    scatter(y_actual, y_pred, 16, 'filled');
    grid on; hold on;

    mn = min([y_actual; y_pred]);
    mx = max([y_actual; y_pred]);
    plot([mn mx], [mn mx], '--', 'LineWidth', 1, 'Color', 'w');

    hold off;    

    xlabel('Фактичні', 'Interpreter','none');
    ylabel('Прогнозовані', 'Interpreter','none');
    title(ttl, 'Interpreter','none');
end