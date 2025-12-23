using System;
using System.Collections.Generic;
using CMM.Ops.MissionControl.Models;

namespace CMM.Ops.MissionControl.Algorithms;

public sealed class AStarPathfinder
{
    public sealed record Options(double RiskPenalty, int MaxNodes = 200_000);

    public List<GridPoint>? FindPath(MapGrid map, GridPoint start, GridPoint goal, Options options)
    {
        if (!map.InBounds(start.X, start.Y) || !map.InBounds(goal.X, goal.Y)) return null;
        if (map[start.X, start.Y].Type == CellType.Obstacle) return null;
        if (map[goal.X, goal.Y].Type == CellType.Obstacle) return null;

        var open = new PriorityQueue<GridPoint, double>();
        var cameFrom = new Dictionary<GridPoint, GridPoint>();
        var gScore = new Dictionary<GridPoint, double>
        {
            [start] = 0
        };

        open.Enqueue(start, Heuristic(start, goal));

        var visited = 0;

        while (open.Count > 0)
        {
            var current = open.Dequeue();
            visited++;
            if (visited > options.MaxNodes) return null;

            if (current.Equals(goal))
                return Reconstruct(cameFrom, current);

            foreach (var n in map.Neighbors4(current))
            {
                if (map[n.X, n.Y].Type == CellType.Obstacle) continue;

                var step = 1.0 + map[n.X, n.Y].Risk * options.RiskPenalty;
                var tentative = gScore[current] + step;

                if (!gScore.TryGetValue(n, out var old) || tentative < old)
                {
                    cameFrom[n] = current;
                    gScore[n] = tentative;
                    var f = tentative + Heuristic(n, goal);
                    open.Enqueue(n, f);
                }
            }
        }

        return null;
    }

    private static double Heuristic(GridPoint a, GridPoint b)
        => Math.Abs(a.X - b.X) + Math.Abs(a.Y - b.Y); // Manhattan

    private static List<GridPoint> Reconstruct(Dictionary<GridPoint, GridPoint> cameFrom, GridPoint current)
    {
        var path = new List<GridPoint> { current };
        while (cameFrom.TryGetValue(current, out var prev))
        {
            current = prev;
            path.Add(current);
        }
        path.Reverse();
        return path;
    }
}
