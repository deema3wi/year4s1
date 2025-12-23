local Env = {}

Env.data = {
    target = {x = 0, y = 0, radius = 20},
    obstacles = {}
}

function Env.load()
    for i = 1, 15 do
        table.insert(Env.data.obstacles, {
            x = love.math.random(100, 700),
            y = love.math.random(50, 500),
            w = love.math.random(30, 80),
            h = love.math.random(30, 80)
        })
    end
    Env.respawnTarget()
end

function Env.respawnTarget()
    local safe = false
    local newX, newY
    local margin = Env.data.target.radius + 5

    while not safe do
        newX = love.math.random(50, 750)
        newY = love.math.random(50, 550)
        safe = true

        for _, obs in ipairs(Env.data.obstacles) do
            if newX > (obs.x - margin) and 
               newX < (obs.x + obs.w + margin) and
               newY > (obs.y - margin) and 
               newY < (obs.y + obs.h + margin) then
                safe = false
                break
            end
        end
    end
    Env.data.target.x = newX
    Env.data.target.y = newY
end

function Env.draw()
    love.graphics.setColor(0.8, 0.3, 0.3)
    for _, obs in ipairs(Env.data.obstacles) do
        love.graphics.rectangle("fill", obs.x, obs.y, obs.w, obs.h)
        love.graphics.setColor(0.9, 0.4, 0.4)
        love.graphics.rectangle("line", obs.x, obs.y, obs.w, obs.h)
        love.graphics.setColor(0.8, 0.3, 0.3)
    end

    love.graphics.setColor(0.2, 0.8, 0.2)
    love.graphics.circle("line", Env.data.target.x, Env.data.target.y, Env.data.target.radius)
    love.graphics.print("TARGET", Env.data.target.x - 20, Env.data.target.y - 35)
end

return Env
