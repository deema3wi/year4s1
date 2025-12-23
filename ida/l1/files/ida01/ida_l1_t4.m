ida_l1_t3

%% TASK 4 =================================================
x1_tr = x_train(:,1);  x2_tr = x_train(:,2);  y_tr = y_train;
x1_te = x_test(:,1);   x2_te = x_test(:,2);   y_te = y_test;

ntr = numel(y_tr);
nte = numel(y_te);

Xtr1 = [ones(ntr,1), x1_tr, x2_tr];
Xtr2 = [ones(ntr,1), x1_tr, x2_tr, x1_tr.^2, x2_tr.^2];
Xtr3 = [ones(ntr,1), x1_tr, x2_tr, x1_tr.^2, x2_tr.^2, x1_tr.*x2_tr];
Xtr4 = [ones(ntr,1), x1_tr, x2_tr, x1_tr.^2, x2_tr.^2, x1_tr.^3, x2_tr.^3];
Xtr5 = [ones(ntr,1), x1_tr, x2_tr, x1_tr.*x2_tr, (x1_tr.^2).*x2_tr, x1_tr.*(x2_tr.^2)];

a1 = Xtr1 \ y_tr;
a2 = Xtr2 \ y_tr;
a3 = Xtr3 \ y_tr;
a4 = Xtr4 \ y_tr;
a5 = Xtr5 \ y_tr;

Xte1 = [ones(nte,1), x1_te, x2_te];
Xte2 = [ones(nte,1), x1_te, x2_te, x1_te.^2, x2_te.^2];
Xte3 = [ones(nte,1), x1_te, x2_te, x1_te.^2, x2_te.^2, x1_te.*x2_te];
Xte4 = [ones(nte,1), x1_te, x2_te, x1_te.^2, x2_te.^2, x1_te.^3, x2_te.^3];
Xte5 = [ones(nte,1), x1_te, x2_te, x1_te.*x2_te, (x1_te.^2).*x2_te, x1_te.*(x2_te.^2)];

yhat_tr1 = Xtr1*a1;  yhat_te1 = Xte1*a1;
yhat_tr2 = Xtr2*a2;  yhat_te2 = Xte2*a2;
yhat_tr3 = Xtr3*a3;  yhat_te3 = Xte3*a3;
yhat_tr4 = Xtr4*a4;  yhat_te4 = Xte4*a4;
yhat_tr5 = Xtr5*a5;  yhat_te5 = Xte5*a5;

[RMSE_tr1, R2_tr1] = metrics(y_tr, yhat_tr1);  [RMSE_te1, R2_te1] = metrics(y_te, yhat_te1);
[RMSE_tr2, R2_tr2] = metrics(y_tr, yhat_tr2);  [RMSE_te2, R2_te2] = metrics(y_te, yhat_te2);
[RMSE_tr3, R2_tr3] = metrics(y_tr, yhat_tr3);  [RMSE_te3, R2_te3] = metrics(y_te, yhat_te3);
[RMSE_tr4, R2_tr4] = metrics(y_tr, yhat_tr4);  [RMSE_te4, R2_te4] = metrics(y_te, yhat_te4);
[RMSE_tr5, R2_tr5] = metrics(y_tr, yhat_tr5);  [RMSE_te5, R2_te5] = metrics(y_te, yhat_te5);

disp('Коефіцієнти моделі 1 (train):'); disp(a1);
disp('Коефіцієнти моделі 2 (train):'); disp(a2);
disp('Коефіцієнти моделі 3 (train):'); disp(a3);
disp('Коефіцієнти моделі 4 (train):'); disp(a4);
disp('Коефіцієнти моделі 5 (train):'); disp(a5);

t4_results = table( ...
    (1:5)', ...
    [3;5;6;7;6], ...
    [RMSE_tr1;RMSE_tr2;RMSE_tr3;RMSE_tr4;RMSE_tr5], ...
    [R2_tr1;R2_tr2;R2_tr3;R2_tr4;R2_tr5], ...
    [RMSE_te1;RMSE_te2;RMSE_te3;RMSE_te4;RMSE_te5], ...
    [R2_te1;R2_te2;R2_te3;R2_te4;R2_te5], ...
    'VariableNames', {'Model','p','RMSE_Train','R2_Train','RMSE_Test','R2_Test'} );

disp(t4_results);

function [rmse, r2] = metrics(y, yhat)
    err = y - yhat;
    rmse = sqrt(mean(err.^2));
    sse = sum(err.^2);
    sst = sum((y - mean(y)).^2);
    r2 = 1 - sse/sst;
end