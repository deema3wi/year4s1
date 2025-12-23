local n = 9
local q = 5

local concept_names = {
  "C1 – Рівень резервного живлення (UPS)",
  "C2 – Резервування Інтернет-каналу",
  "C3 – Моніторинг і алерти",
  "C4 – Оптимізація Nginx/кешування",
  "C5 – Планові техобслуговування",
  "C6 – Доступність сервісу (uptime)",
  "C7 – Швидкодія/затримка (latency)",
  "C8 – Задоволеність користувачів",
  "C9 – Витратна ефективність (чим більше – тим краще)",
}

local W = {
  {0, 0, 0, 0, 0, 0.75, 0.25, 0.25, -0.75},
  {0, 0, 0, 0, 0, 0.75, 0.5, 0.25, -0.75},
  {0, 0, 0, 0, 0, 0.5, 0, 0.5, 0.5},
  {0, 0, 0, 0, 0, 0.25, 0.75, 0.5, 0.25},
  {0, 0, 0, 0, 0, 0.5, 0, 0.25, -0.25},
  {0, 0, 0, 0, 0, 0, 0, 0.75, 0},
  {0, 0, 0, 0, 0, 0, 0, 0.5, 0},
  {0, 0, 0, 0, 0.25, 0, 0, 0, 0},
  {0, 0, 0.25, 0, 0.25, 0, 0, 0, 0},
}

local scenarios = {
  { name = "S1 (Економний базовий)", u = {0.2, 0.1, 0.3, 0.2, 0.2} },
  { name = "S2 (Збалансований)", u = {0.5, 0.4, 0.5, 0.5, 0.4} },
  { name = "S3 (Фокус на швидкодію)", u = {0.4, 0.3, 0.5, 0.9, 0.4} },
  { name = "S4 (Фокус на надійність)", u = {0.9, 0.8, 0.6, 0.5, 0.7} },
  { name = "S5 (Моніторинг + ТО)", u = {0.6, 0.3, 0.9, 0.4, 0.9} },
  { name = "S6 (Ультра-дешевий)", u = {0.2, 0.1, 0.1, 0.1, 0.1} },
  { name = "S7 (Преміум)", u = {0.9, 0.9, 0.9, 0.9, 0.9} },
  { name = "S8 (Розумна оптимізація)", u = {0.3, 0.3, 0.9, 0.9, 0.4} },
}

local function zeros(m)
  local v = {}
  for i = 1, m do v[i] = 0 end
  return v
end

local function vec_sub(a, b)
  local r = {}
  for i = 1, #a do r[i] = a[i] - b[i] end
  return r
end

local function vec_add(a, b)
  local r = {}
  for i = 1, #a do r[i] = a[i] + b[i] end
  return r
end

local function vec_mul_row_mat(v, M)
  local r = {}
  for j = 1, #M[1] do
    local s = 0
    for i = 1, #v do
      s = s + v[i] * M[i][j]
    end
    r[j] = s
  end
  return r
end

local function max_abs_diff(a, b)
  local m = 0
  for i = 1, #a do
    local d = math.abs(a[i] - b[i])
    if d > m then m = d end
  end
  return m
end

local function simulate_stationary(X0, eps, max_iter)
  local X_prev = zeros(n)
  local X_curr = {}
  for i = 1, n do X_curr[i] = X0[i] end
  local it = 0
  while it < max_iter do
    it = it + 1
    local delta = vec_sub(X_curr, X_prev)
    local influence = vec_mul_row_mat(delta, W)
    local X_next = vec_add(X_curr, influence)
    if max_abs_diff(X_next, X_curr) < eps then
      return X_next, it
    end
    X_prev, X_curr = X_curr, X_next
  end
  return X_curr, it
end

local function clamp(x, lo, hi)
  if x < lo then return lo end
  if x > hi then return hi end
  return x
end

local eps = 1e-9
local max_iter = 5000
local station = {}
for si = 1, #scenarios do
  local X0 = zeros(n)
  for i = 1, q do X0[i] = scenarios[si].u[i] end
  local Xl, it = simulate_stationary(X0, eps, max_iter)
  station[si] = { Xl = Xl, it = it }
end

local max_pos = zeros(n)
local min_neg = zeros(n)
for i = 1, n do
  max_pos[i] = -math.huge
  min_neg[i] = math.huge
end
for si = 1, #scenarios do
  local Xl = station[si].Xl
  for i = 1, n do
    if Xl[i] > max_pos[i] then max_pos[i] = Xl[i] end
    if Xl[i] < min_neg[i] then min_neg[i] = Xl[i] end
  end
end

local function normalize_x(i, x)
  if x > 0 then
    local d = max_pos[i]
    if d == 0 then return 0 end
    return clamp(x / d, -1, 1)
  elseif x < 0 then
    local d = min_neg[i] -- negative
    if d == 0 then return 0 end
    return clamp((-x) / d, -1, 1)
  else
    return 0
  end
end

local function mu_perfection(x_hat)
  return (x_hat + 1) / 2
end

local results = {}
for si = 1, #scenarios do
  local Xl = station[si].Xl
  local mu = {}
  local muD = 1
  for i = q + 1, n do
    local x_hat = normalize_x(i, Xl[i])
    local m = mu_perfection(x_hat)
    mu[i] = m
    if m < muD then muD = m end
  end
  results[si] = { name = scenarios[si].name, mu = mu, muD = muD, it = station[si].it }
end

table.sort(results, function(a,b) return a.muD > b.muD end)

print('Ранжування сценаріїв за принципом Беллмана–Заде (μD = min μi):')
print(string.format('%-28s  μ6      μ7      μ8      μ9      μD     it', 'Сценарій'))
for _, r in ipairs(results) do
  local mu6 = r.mu[6] or 0
  local mu7 = r.mu[7] or 0
  local mu8 = r.mu[8] or 0
  local mu9 = r.mu[9] or 0
  print(string.format('%-28s  %.6f  %.6f  %.6f  %.6f  %.6f  %d', r.name, mu6, mu7, mu8, mu9, r.muD, r.it))
end

local best = results[1]
print('')
print('Найкращий сценарій: ' .. best.name)
print(string.format('μD = %.6f', best.muD))
