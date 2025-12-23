% TASK 6 ==================================================
if ~exist('T','var') || ~exist('trainIdx','var')
    ida_l1_data;
end

t6 = ida_task6_scan(T, trainIdx, testIdx);

disp('TOP-10 за RMSE_Test:');
disp(t6.t6_sorted(1:min(10,height(t6.t6_sorted)), :));

ida_plot_yhat(t6.best.ytr, t6.best.yhat_tr, t6.best.VnName + ", " + t6.best.ModelName + " (Train)");
ida_plot_yhat(t6.best.yte, t6.best.yhat_te, t6.best.VnName + ", " + t6.best.ModelName + " (Test)");
