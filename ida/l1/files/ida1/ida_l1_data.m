% DATA ===================================================
FILE = '/MATLAB Drive/Examples/Residential-Building-Data-Set.xlsx';
T = readtable(FILE, 'Sheet', 'Data', 'NumHeaderLines', 1);

x1 = T.V_5;  x1 = x1(:);
x2 = T.V_13; x2 = x2(:);
y  = T.V_9;  y  = y(:);

N = height(T);
[trainIdx, testIdx] = ida_holdout_split(N, 0.7, 1);

x_train = [x1(trainIdx), x2(trainIdx)];
y_train = y(trainIdx);

x_test = [x1(testIdx), x2(testIdx)];
y_test = y(testIdx);
