ida_l1_t2

%% TASK 3 =================================================
figure;
scatter(x1, y, 'filled');
grid on;
xlabel('Попередньо оцінена вартість будівництва');
ylabel('Фактична ціна продажу');
title('Залежність V-9 від V-5');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

figure;
scatter(x2, y, 'filled');
grid on;
xlabel('Індекс оптових цін на буд. матеріали');
ylabel('Фактична ціна продажу');
title('Залежність V-9 від V13');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);