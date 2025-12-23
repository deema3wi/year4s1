-- Лабораторна робота №3: Модель ПЗ БПЛА з уникненням перешкод
-- Спеціальність 122 КН, КММ

function love.load()
  love.window.setTitle("Lab 3: Drone AI Simulation")
  love.window.setMode(800, 600)

  -- Параметри симуляції
  simulation = {
    target = {x = 750, y = 500, radius = 20},
    obstacles = {}
  }

  -- Генерація випадкових перешкод (імітація міського середовища/зон)
  for i = 1, 15 do
    table.insert(simulation.obstacles, {
      x = love.math.random(100, 700),
      y = love.math.random(50, 500),
      w = love.math.random(30, 80),
      h = love.math.random(30, 80)
    })
  end

  respawnTarget()

  -- Модель БПЛА (Агент)
  drone = {
    x = 50,
    y = 50,
    angle = 0,
    speed = 150,
    radius = 10,
    sensorRange = 60, -- БУЛО 100. Зменшуємо до 60, щоб він був "сміливішим"
    sensors = { -0.5, -0.25, 0, 0.25, 0.5 }
  }
end

-- Допоміжна функція: перевірка перетину лінії та прямокутника
function checkIntersection(x1, y1, x2, y2, rect)
  -- Проста перевірка AABB для променів (спрощена)
  -- Для точної фізики зазвичай використовують складнішу геометрію,
  -- але для моделі поведінки достатньо перевірити кінцеву точку або крок.
  local steps = 20
  for i = 0, steps do
    local t = i / steps
    local px = x1 + (x2 - x1) * t
    local py = y1 + (y2 - y1) * t
    if px > rect.x and px < rect.x + rect.w and py > rect.y and py < rect.y + rect.h then
      return true, t -- повертаємо факт зіткнення і відстань (0..1)
    end
  end
  return false, 1
end

function love.update(dt)
  -- 1. СПОЧАТКУ РОЗРАХУНОК ЦІЛІ (Перенесли нагору, бо це треба знати раніше)
  local dx = simulation.target.x - drone.x
  local dy = simulation.target.y - drone.y
  local distToTarget = math.sqrt(dx*dx + dy*dy)
  local targetAngle = math.atan2(dy, dx)

  -- Якщо дійшли до цілі - респаун
  if distToTarget < 20 then
    respawnTarget()
    -- Оновлюємо змінні для нової цілі, щоб уникнути ривка в цьому кадрі
    dx = simulation.target.x - drone.x
    dy = simulation.target.y - drone.y
    distToTarget = math.sqrt(dx*dx + dy*dy)
  end

  -- 2. ЛОГІКА СЕНСОРІВ
  local avoidTurn = 0
  local detectionCount = 0
  local closestObstacleDist = 1.0 -- 1.0 означає "максимально далеко" (на межі сенсора)

  for _, sensorAngle in ipairs(drone.sensors) do
    local globalAngle = drone.angle + sensorAngle
    local ex = drone.x + math.cos(globalAngle) * drone.sensorRange
    local ey = drone.y + math.sin(globalAngle) * drone.sensorRange

    local hit = false
    local distFactor = 1

    for _, obs in ipairs(simulation.obstacles) do
      local isHit, t = checkIntersection(drone.x, drone.y, ex, ey, obs)
      if isHit then
        hit = true
        if t < distFactor then distFactor = t end
      end
    end

    if hit then
      -- Запам'ятовуємо найближчу перешкоду
      if distFactor < closestObstacleDist then closestObstacleDist = distFactor end

      -- Розрахунок сили відштовхування
      local force = (1.0 - distFactor) * 8 -- Збільшили коефіцієнт сили

      if sensorAngle < 0 then
        avoidTurn = avoidTurn + force
      elseif sensorAngle > 0 then
        avoidTurn = avoidTurn - force
      else
        avoidTurn = avoidTurn + (love.math.random() > 0.5 and 5 or -5)
      end
      detectionCount = detectionCount + 1
    end
  end

  -- 3. КОНТРОЛЕР ПОЛЬОТУ (Smart Blending)
  local steerSpeed = 4 * dt

  -- Кут до цілі
  local angleDiff = targetAngle - drone.angle
  while angleDiff > math.pi do angleDiff = angleDiff - 2*math.pi end
  while angleDiff < -math.pi do angleDiff = angleDiff + 2*math.pi end

  -- === ГОЛОВНЕ ВИПРАВЛЕННЯ ===
  -- Логіка пріоритетів: 
  -- Якщо ми далеко від цілі -> Уникаємо перешкод пріоритетно.
  -- Якщо ми БЛИЗЬКО до цілі -> Ігноруємо перешкоди (ризикуємо), якщо тільки це не лобове зіткнення.

  local avoidanceWeight = 1.0

  -- Якщо до цілі менше 100 пікселів, поступово зменшуємо страх перед перешкодами
  if distToTarget < 100 then
    avoidanceWeight = distToTarget / 100 -- Від 0.0 до 1.0
  end

  -- Але якщо перешкода КРИТИЧНО близько (менше 15% довжини сенсора), паніка повертається на максимум
  if closestObstacleDist < 0.15 then
    avoidanceWeight = 2.0 -- Екстрене ухилення
  end

  if detectionCount > 0 then
    -- Змішуємо бажання повернути до цілі і бажання ухилитися
    -- avoidTurn штовхає вбік від стіни, angleDiff тягне до цілі
    local finalTurn = (avoidTurn * avoidanceWeight) + (angleDiff * 0.5) 
    drone.angle = drone.angle + finalTurn * dt
  else
    -- Чистий політ до цілі
    drone.angle = drone.angle + angleDiff * steerSpeed
  end

  -- Оновлення позиції
  drone.x = drone.x + math.cos(drone.angle) * drone.speed * dt
  drone.y = drone.y + math.sin(drone.angle) * drone.speed * dt

  -- Обмеження екрану
  if drone.x < 0 then drone.x = 0 elseif drone.x > 800 then drone.x = 800 end
  if drone.y < 0 then drone.y = 0 elseif drone.y > 600 then drone.y = 600 end
end
function love.draw()
  -- Малювання фону
  love.graphics.clear(0.1, 0.1, 0.15) -- Темний фон (імітація нічного польоту/інтерфейсу оператора)

  -- Малювання перешкод
  love.graphics.setColor(0.8, 0.3, 0.3)
  for _, obs in ipairs(simulation.obstacles) do
    love.graphics.rectangle("fill", obs.x, obs.y, obs.w, obs.h)
    love.graphics.setColor(0.9, 0.4, 0.4)
    love.graphics.rectangle("line", obs.x, obs.y, obs.w, obs.h)
    love.graphics.setColor(0.8, 0.3, 0.3)
  end

  -- Малювання цілі
  love.graphics.setColor(0.2, 0.8, 0.2)
  love.graphics.circle("line", simulation.target.x, simulation.target.y, simulation.target.radius)
  love.graphics.print("TARGET", simulation.target.x - 20, simulation.target.y - 35)

  -- Малювання сенсорів (Debug View)
  for _, sensorAngle in ipairs(drone.sensors) do
    local globalAngle = drone.angle + sensorAngle
    local ex = drone.x + math.cos(globalAngle) * drone.sensorRange
    local ey = drone.y + math.sin(globalAngle) * drone.sensorRange

    -- Перевірка для кольору променя
    local hit = false
    for _, obs in ipairs(simulation.obstacles) do
      if checkIntersection(drone.x, drone.y, ex, ey, obs) then hit = true break end
    end

    if hit then
      love.graphics.setColor(1, 0, 0, 0.7) -- Червоний якщо бачить перешкоду
    else
      love.graphics.setColor(0, 1, 1, 0.2) -- Блакитний сканер
    end
    love.graphics.line(drone.x, drone.y, ex, ey)
  end

  -- Малювання БПЛА
  love.graphics.push()
  love.graphics.translate(drone.x, drone.y)
  love.graphics.rotate(drone.angle)

  love.graphics.setColor(1, 1, 1)
  -- Трикутник (тіло дрона)
  love.graphics.polygon("fill", 10, 0, -10, -7, -10, 7)
  love.graphics.pop()

  -- Інтерфейс (HUD)
  love.graphics.setColor(1, 1, 1)
  love.graphics.print("Lab 3: AI UAV Simulation (Lua/Love2D)", 10, 10)
  love.graphics.print("Status: " .. (drone.x > 0 and "AUTO FLIGHT" or "CRASH"), 10, 30)
  love.graphics.print("Obstacles Detected: " .. "Sensor Active", 10, 50)
end


-- Функція для безпечного спавну цілі
function respawnTarget()
  local safe = false
  local newX, newY
  local margin = simulation.target.radius + 5 -- Радіус цілі + 5 пікселів запасу

  -- Цикл триватиме, доки не знайдемо вільне місце
  while not safe do
    -- Генеруємо випадкові координати з відступом від країв екрану
    newX = love.math.random(50, 750)
    newY = love.math.random(50, 550)
    safe = true -- Припускаємо, що місце безпечне

    -- Перевіряємо кожну перешкоду
    for _, obs in ipairs(simulation.obstacles) do
      -- Перевірка AABB (Axis-Aligned Bounding Box)
      -- Ми розширюємо зону перешкоди на радіус цілі (margin).
      -- Якщо центр цілі потрапляє в цю розширену зону - це колізія.
      if newX > (obs.x - margin) and 
        newX < (obs.x + obs.w + margin) and
        newY > (obs.y - margin) and 
        newY < (obs.y + obs.h + margin) then

        safe = false -- Місце зайняте, пробуємо знову
        break
      end
    end
  end

  -- Застосовуємо безпечні координати
  simulation.target.x = newX
  simulation.target.y = newY
end
