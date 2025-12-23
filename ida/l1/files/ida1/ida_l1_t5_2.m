% TASK 5 ==================================================
if ~exist('T','var') || ~exist('x_train','var')
    ida_l1_data;
end
if ~exist('t4','var')
    t4 = ida_fit_models2(x_train, y_train, x_test, y_test);
end

ida_plot_yhat(t4.y_tr, t4.yhat_tr{1}, 'Лінійна (Train)');
ida_plot_yhat(t4.y_te, t4.yhat_te{1}, 'Лінійна (Test)');

ida_plot_yhat(t4.y_tr, t4.yhat_tr{2}, 'Квадратична (Train)');
ida_plot_yhat(t4.y_te, t4.yhat_te{2}, 'Квадратична (Test)');
