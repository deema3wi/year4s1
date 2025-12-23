% TASK 7 ==================================================
if ~exist('T','var') || ~exist('trainIdx','var')
    ida_l1_data;
end

t7 = ida_fit_linear_all_numeric(T, 'V_9', trainIdx, testIdx);
disp(t7.summary);

ida_plot_yhat(t7.ytr, t7.yhat_tr, 'Лінійна (усі кількісні) (Train)');
ida_plot_yhat(t7.yte, t7.yhat_te, 'Лінійна (усі кількісні) (Test)');
