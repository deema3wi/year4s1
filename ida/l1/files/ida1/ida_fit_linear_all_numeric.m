function out = ida_fit_linear_all_numeric(T, yName, trainIdx, testIdx)
vars = T.Properties.VariableNames;

y = T{:, yName};
y = y(:);

isNum = varfun(@isnumeric, T, 'OutputFormat','uniform');
predNames = vars(isNum);
predNames = setdiff(predNames, {yName}, 'stable');

Xall = double(T{:, predNames});

Xtr0 = Xall(trainIdx, :);
Xte0 = Xall(testIdx,  :);
ytr0 = y(trainIdx); ytr0 = ytr0(:);
yte0 = y(testIdx);  yte0 = yte0(:);

mtr = ida_mask_finite(ytr0, Xtr0);
mte = ida_mask_finite(yte0, Xte0);

Xtr = Xtr0(mtr, :);
ytr = ytr0(mtr);
Xte = Xte0(mte, :);
yte = yte0(mte);

s = std(Xtr, 0, 1);
keep = isfinite(s) & (s > 0);

Xtr = Xtr(:, keep);
Xte = Xte(:, keep);
predKept = predNames(keep);

XtrD = [ones(size(Xtr,1),1), Xtr];
XteD = [ones(size(Xte,1),1), Xte];

a = lsqminnorm(XtrD, ytr);

yhat_tr = XtrD * a;
yhat_te = XteD * a;

[rmse_tr, r2_tr] = ida_metrics(ytr, yhat_tr);
[rmse_te, r2_te] = ida_metrics(yte, yhat_te);

out.ytr = ytr;
out.yte = yte;
out.yhat_tr = yhat_tr;
out.yhat_te = yhat_te;
out.a = a;
out.predNames = predKept;
out.p = size(XtrD,2);
out.rmse_tr = rmse_tr;
out.r2_tr = r2_tr;
out.rmse_te = rmse_te;
out.r2_te = r2_te;
out.summary = table(out.p, rmse_tr, r2_tr, rmse_te, r2_te, ...
    'VariableNames', {'p','RMSE_Train','R2_Train','RMSE_Test','R2_Test'});
end
