% TASK 2 ==================================================
if ~exist('T','var') || ~exist('trainIdx','var') || ~exist('x_train','var')
    ida_l1_data;
end

S = ida_split_stats([x1, x2, y], trainIdx, testIdx, ["V5","V13","V9"]);
disp(S);

ida_plot_split_scatter(x_train(:,1), y_train, x_test(:,1), y_test, ...
    'V5', 'V9', 'Train/Test: V9 vs V5');

ida_plot_split_scatter(x_train(:,2), y_train, x_test(:,2), y_test, ...
    'V13', 'V9', 'Train/Test: V9 vs V13');
