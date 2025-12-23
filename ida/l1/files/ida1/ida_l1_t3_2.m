% TASK 3 ==================================================
if ~exist('T','var') || ~exist('x1','var')
    ida_l1_data;
end

ida_plot_scatter(x1, y, 'V5', 'V9', 'V9 vs V5');
ida_plot_scatter(x2, y, 'V13', 'V9', 'V9 vs V13');
