% TASK 4 ==================================================
if ~exist('T','var') || ~exist('x_train','var')
    ida_l1_data;
end

t4 = ida_fit_models2(x_train, y_train, x_test, y_test);
disp(t4.results);
