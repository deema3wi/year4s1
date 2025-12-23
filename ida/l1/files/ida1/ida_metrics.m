function [rmse, r2] = ida_metrics(y, yhat)
y = y(:);
yhat = yhat(:);
m = isfinite(y) & isfinite(yhat);
y = y(m);
yhat = yhat(m);

err = y - yhat;
rmse = sqrt(mean(err.^2));

sse = sum(err.^2);
sst = sum((y - mean(y)).^2);
r2 = 1 - sse/sst;
end
