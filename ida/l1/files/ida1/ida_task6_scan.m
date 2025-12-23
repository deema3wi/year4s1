function out = ida_task6_scan(T, trainIdx, testIdx)
vars = T.Properties.VariableNames;

x1 = T.V_5; x1 = x1(:);
x2 = T.V_13; x2 = x2(:);
y  = T.V_9; y  = y(:);

isNum = varfun(@isnumeric, T, 'OutputFormat','uniform');
cand = vars(isNum);
cand = setdiff(cand, {'V_5','V_13','V_9'}, 'stable');

modelNames = ["M1 linear", "M2 quad", "M3 full quad", "M4 cubic", "M5 mixed cubic"];

rows = numel(cand) * 5;
VnName = strings(rows,1);
ModelId = zeros(rows,1);
ModelName = strings(rows,1);
p = zeros(rows,1);
RMSE_Train = zeros(rows,1);
R2_Train = zeros(rows,1);
RMSE_Test = zeros(rows,1);
R2_Test = zeros(rows,1);

bestRMSE = inf;
best = struct();

r = 0;

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
        Xtr = ida_designMatrix3(m, x1tr, x2tr, x3tr);
        Xte = ida_designMatrix3(m, x1te, x2te, x3te);

        a = lsqminnorm(Xtr, ytr);

        yhat_tr = Xtr * a;
        yhat_te = Xte * a;

        [rmse_tr, r2_tr] = ida_metrics(ytr, yhat_tr);
        [rmse_te, r2_te] = ida_metrics(yte, yhat_te);

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
            best.VnName = string(cand{i});
            best.ModelId = m;
            best.ModelName = modelNames(m);
            best.p = size(Xtr,2);
            best.a = a;
            best.ytr = ytr; best.yhat_tr = yhat_tr;
            best.yte = yte; best.yhat_te = yhat_te;
        end
    end
end

t6 = table(VnName, ModelId, ModelName, p, RMSE_Train, R2_Train, RMSE_Test, R2_Test);
t6_sorted = sortrows(t6, 'RMSE_Test', 'ascend');

out.t6 = t6;
out.t6_sorted = t6_sorted;
out.best = best;
end
