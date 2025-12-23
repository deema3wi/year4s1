function out = ida_fit_models2(x_train, y_train, x_test, y_test)
x1_tr = x_train(:,1); x2_tr = x_train(:,2); y_tr = y_train(:);
x1_te = x_test(:,1);  x2_te = x_test(:,2);  y_te = y_test(:);

modelNames = ["M1 linear", "M2 quad", "M3 full quad", "M4 cubic", "M5 mixed cubic"];
p = zeros(5,1);
RMSE_Train = zeros(5,1);
R2_Train = zeros(5,1);
RMSE_Test = zeros(5,1);
R2_Test = zeros(5,1);

a = cell(5,1);
yhat_tr = cell(5,1);
yhat_te = cell(5,1);

for m = 1:5
    Xtr = ida_designMatrix2(m, x1_tr, x2_tr);
    Xte = ida_designMatrix2(m, x1_te, x2_te);

    a{m} = lsqminnorm(Xtr, y_tr);
    p(m) = size(Xtr,2);

    yhat_tr{m} = Xtr * a{m};
    yhat_te{m} = Xte * a{m};

    [RMSE_Train(m), R2_Train(m)] = ida_metrics(y_tr, yhat_tr{m});
    [RMSE_Test(m),  R2_Test(m)]  = ida_metrics(y_te, yhat_te{m});
end

results = table((1:5)', modelNames', p, RMSE_Train, R2_Train, RMSE_Test, R2_Test, ...
    'VariableNames', {'Model','Name','p','RMSE_Train','R2_Train','RMSE_Test','R2_Test'});

out.modelNames = modelNames;
out.a = a;
out.yhat_tr = yhat_tr;
out.yhat_te = yhat_te;
out.results = results;
out.y_tr = y_tr;
out.y_te = y_te;
end
