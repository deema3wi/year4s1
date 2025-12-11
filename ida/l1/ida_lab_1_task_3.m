figure;
scatter(x1, y, 'filled');
grid on;
xlabel('V5 – попередньо оцінена вартість будівництва (10000 IRRm)');
ylabel('V9 – фактична ціна продажу (10000 IRRm)');
title('Однофакторна залежність y = f(x_1 = V5)');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

figure;
scatter(x2, y, 'filled');
grid on;
xlabel('V13 – економічний індекс (V-13)');
ylabel('V9 – фактична ціна продажу (10000 IRRm)');
title('Однофакторна залежність y = f(x_2 = V13)');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);