local x = {0, 1, 2}
local y = {2, 5, 5}
local x_pred = 3

local n = #x

local sumX, sumY, sumXX, sumXY = 0, 0, 0, 0
for i = 1, n do
  sumX  = sumX  + x[i]
  sumY  = sumY  + y[i]
  sumXX = sumXX + x[i] * x[i]
  sumXY = sumXY + x[i] * y[i]
end

local denom = n * sumXX - sumX * sumX
if denom == 0 then
  error("Denominator is zero (all x values identical) – cannot fit a line.")
end

local b = (n * sumXY - sumX * sumY) / denom
local a = (sumY - b * sumX) / n

local y_pred = a + b * x_pred

print(string.format("n = %d", n))
print(string.format("Σx  = %.6f", sumX))
print(string.format("Σy  = %.6f", sumY))
print(string.format("Σx² = %.6f", sumXX))
print(string.format("Σxy = %.6f", sumXY))
print(string.format("a = %.6f", a))
print(string.format("b = %.6f", b))
print(string.format("Forecast: y(%d) = %.6f", x_pred, y_pred))
