local Utils = require("utils")
local Drone = {}

Drone.body = {
    x = 50,
    y = 50,
    angle = 0,
    speed = 150,
    radius = 10,
    sensorRange = 60,
    sensors = { -0.5, -0.25, 0, 0.25, 0.5 }
}

function Drone.update(dt, target, obstacles)
    local d = Drone.body
    
    local dx = target.x - d.x
    local dy = target.y - d.y
    local distToTarget = math.sqrt(dx*dx + dy*dy)
    local targetAngle = math.atan2(dy, dx)

    local avoidTurn = 0
    local detectionCount = 0
    local closestObstacleDist = 1.0 

    for _, sensorAngle in ipairs(d.sensors) do
        local globalAngle = d.angle + sensorAngle
        local ex = d.x + math.cos(globalAngle) * d.sensorRange
        local ey = d.y + math.sin(globalAngle) * d.sensorRange

        local hit = false
        local distFactor = 1.0

        for _, obs in ipairs(obstacles) do
            local isHit, t = Utils.checkIntersection(d.x, d.y, ex, ey, obs, d.radius)
            if isHit then
                hit = true
                if t < distFactor then distFactor = t end
            end
        end

        if hit then
            if distFactor < closestObstacleDist then closestObstacleDist = distFactor end
            local force = (1.0 - distFactor) * 8
            
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

    local steerSpeed = 4 * dt
    local angleDiff = targetAngle - d.angle
    
    while angleDiff > math.pi do angleDiff = angleDiff - 2*math.pi end
    while angleDiff < -math.pi do angleDiff = angleDiff + 2*math.pi end

    local avoidanceWeight = 1.0

    if distToTarget < 100 then
        avoidanceWeight = distToTarget / 100
    end

    if closestObstacleDist < 0.15 then
        avoidanceWeight = 2.0
    end

    if detectionCount > 0 then
        local finalTurn = (avoidTurn * avoidanceWeight) + (angleDiff * 0.5) 
        d.angle = d.angle + finalTurn * dt
    else
        d.angle = d.angle + angleDiff * steerSpeed
    end

    d.x = d.x + math.cos(d.angle) * d.speed * dt
    d.y = d.y + math.sin(d.angle) * d.speed * dt

    if d.x < 0 then d.x = 0 elseif d.x > 800 then d.x = 800 end
    if d.y < 0 then d.y = 0 elseif d.y > 600 then d.y = 600 end

    return distToTarget
end

function Drone.draw(obstacles)
    local d = Drone.body
    
    for _, sensorAngle in ipairs(d.sensors) do
        local globalAngle = d.angle + sensorAngle
        local ex = d.x + math.cos(globalAngle) * d.sensorRange
        local ey = d.y + math.sin(globalAngle) * d.sensorRange

        local hit = false
        for _, obs in ipairs(obstacles) do
            if Utils.checkIntersection(d.x, d.y, ex, ey, obs) then hit = true break end
        end

        if hit then love.graphics.setColor(1, 0, 0, 0.7)
        else love.graphics.setColor(0, 1, 1, 0.2) end
        love.graphics.line(d.x, d.y, ex, ey)
    end

    love.graphics.push()
    love.graphics.translate(d.x, d.y)
    love.graphics.rotate(d.angle)
    love.graphics.setColor(1, 1, 1)
    love.graphics.polygon("fill", 10, 0, -10, -7, -10, 7)
    love.graphics.pop()
end

return Drone
