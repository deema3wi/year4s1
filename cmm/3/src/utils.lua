local Utils = {}

function Utils.checkIntersection(x1, y1, x2, y2, rect, padding)
    local pad = padding or 0

    local rx = rect.x - pad
    local ry = rect.y - pad
    local rw = rect.w + pad * 2
    local rh = rect.h + pad * 2

    local steps = 20
    
    for i = 0, steps do
        local t = i / steps
        local px = x1 + (x2 - x1) * t
        local py = y1 + (y2 - y1) * t
        if px > rx and px < rx + rw and py > ry and py < ry + rh then
            return true, t
        end
    end
    return false, 1
end

return Utils
