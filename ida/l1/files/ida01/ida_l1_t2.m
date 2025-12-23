T = readtable('/MATLAB Drive/Examples/Residential-Building-Data-Set.xlsx', 'Sheet', 'Data', 'NumHeaderLines', 1);

x1 = T.V_5;
x2 = T.V_13;
y  = T.V_9;

xy = [x1, x2, y];
N  = height(T);

trainRatio = 0.7;
rng(1);
idx = randperm(N);
Ntrain = round(trainRatio * N);
trainIdx = idx(1:Ntrain);
testIdx  = idx(Ntrain+1:end);

x_train = xy(trainIdx, 1:2);
y_train = xy(trainIdx, 3);

x_test = xy(testIdx, 1:2);
y_test = xy(testIdx, 3);

all   = xy;
train = xy(trainIdx, :);
test  = xy(testIdx, :);

mean_all   = mean(all);
mean_train = mean(train);
mean_test  = mean(test);

std_all   = std(all);
std_train = std(train);
std_test  = std(test);

min_all   = min(all);
min_train = min(train);
min_test  = min(test);

max_all   = max(all);
max_train = max(train);
max_test  = max(test);

rowNames = {'Mean','Std','Min','Max'};

stats_x1 = array2table([mean_all(1); std_all(1); min_all(1); max_all(1)], ...
    'RowNames', rowNames, 'VariableNames', {'All'});
stats_x1.Train = [mean_train(1); std_train(1); min_train(1); max_train(1)];
stats_x1.Test  = [mean_test(1);  std_test(1);  min_test(1);  max_test(1)];

stats_x2 = array2table([mean_all(2); std_all(2); min_all(2); max_all(2)], ...
    'RowNames', rowNames, 'VariableNames', {'All'});
stats_x2.Train = [mean_train(2); std_train(2); min_train(2); max_train(2)];
stats_x2.Test  = [mean_test(2);  std_test(2);  min_test(2);  max_test(2)];

stats_y = array2table([mean_all(3); std_all(3); min_all(3); max_all(3)], ...
    'RowNames', rowNames, 'VariableNames', {'All'});
stats_y.Train = [mean_train(3); std_train(3); min_train(3); max_train(3)];
stats_y.Test  = [mean_test(3);  std_test(3);  min_test(3);  max_test(3)];

% OUTPUT =================================================
disp('Статистика для x1 (V5)'); disp(stats_x1);
disp('Статистика для x2 (V13)'); disp(stats_x2);
disp('Статистика для y (V9)');  disp(stats_y);

x1_train = x_train(:,1); x1_test = x_test(:,1);
x2_train = x_train(:,2); x2_test = x_test(:,2);

figure;
hold on;
scatter(x1_train, y_train, 20, 'magenta', 'filled');
scatter(x1_test,  y_test,  20, 'cyan',    'filled');
hold off;
grid on;
xlabel('Попередньо оцінена вартість будівництва');
ylabel('Фактична ціна продажу');
title('Розподіл на Train і Test');
legend('Train','Test','Location','best');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

figure;
hold on;
scatter(x2_train, y_train, 20, 'magenta', 'filled');
scatter(x2_test,  y_test,  20, 'cyan',    'filled');
hold off;
grid on;
xlabel('Індекс оптових цін на буд. матеріали');
ylabel('Фактична ціна продажу');
title('Розподіл на Train і Test');
legend('Train','Test','Location','best');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);