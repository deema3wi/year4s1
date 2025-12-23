using System;
using System.Collections.Generic;
using CMM.Ops.MissionControl.Models;

namespace CMM.Ops.MissionControl.Algorithms;

/// <summary>
/// Monte-Carlo strategy selection: evaluates a few candidate routes (different risk penalties)
/// by simulating random failures/damage on risky cells.
/// </summary>
public sealed class MonteCarloPlanner
{
    public sealed record Candidate(string Name, double RiskPenalty);

    public sealed record Choice(Candidate Candidate, List<GridPoint> Path, double Score, double SuccessRate);

    private readonly AStarPathfinder _pathfinder = new();

    public Choice? ChooseBest(MapGrid map, GridPoint start, GridPoint goal, double crewRiskMultiplier, SimulationTuning? tuning = null, int simulationsPerCandidate = 120, int seed = 0)
    {
        var rng = seed == 0 ? new Random() : new Random(seed);

        var candidates = new[]
        {
            new Candidate("Fast", 0.4),
            new Candidate("Balanced", 2.0),
            new Candidate("Safe", 4.5),
        };

        Choice? best = null;

        foreach (var c in candidates)
        {
            var path = _pathfinder.FindPath(map, start, goal, new AStarPathfinder.Options(c.RiskPenalty));
            if (path is null || path.Count < 2) continue;

            var successes = 0;
            var totalScore = 0.0;

            for (var i = 0; i < simulationsPerCandidate; i++)
            {
                var score = SimulateOnce(map, path, crewRiskMultiplier, tuning, rng);
                if (score > 0) successes++;
                totalScore += score;
            }

            var successRate = successes / (double)simulationsPerCandidate;
            var avgScore = totalScore / simulationsPerCandidate;

            // Prefer higher average score; tie-breaker: success rate.
            if (best is null || avgScore > best.Score || (Math.Abs(avgScore - best.Score) < 1e-6 && successRate > best.SuccessRate))
                best = new Choice(c, path, avgScore, successRate);
        }

        return best;
    }

    private static double SimulateOnce(MapGrid map, List<GridPoint> path, double crewRiskMultiplier, SimulationTuning? tuning, Random rng)
    {
        var t = tuning ?? new SimulationTuning();

        // Returns positive score for success, negative for failure
        var battery = t.StartBatteryPct;
        var health = t.StartHealthPct;
        var riskSum = 0.0;

        for (var i = 1; i < path.Count; i++)
        {
            var p = path[i];
            var cell = map[p.X, p.Y];

            battery -= 0.25 + cell.Risk * t.RiskBatteryDrain; // per step baseline + risk drain
            riskSum += cell.Risk;

            var risk = Math.Clamp(cell.Risk * t.RiskMultiplier * crewRiskMultiplier, 0, 1);

            // Random incident probability grows with risk (mirrors MissionSimulator tuning)
            var incidentProb = Math.Clamp(t.BaseIncidentChance + risk * t.RiskIncidentFactor, 0, 0.95);
            if (rng.NextDouble() < incidentProb)
            {
                var baseDmg = t.MinDamage + rng.NextDouble() * Math.Max(0.0, t.MaxDamage - t.MinDamage);
                var dmg = baseDmg * (0.60 + risk * 0.80) * t.DamageMultiplier;
                health -= dmg;
                battery -= t.ExtraBatteryOnIncident;
            }

            if (battery <= 0 || health <= 0)
                return -100;
        }

        // Score prefers short paths and low total risk
        var len = path.Count;
        var score = 1000 - len * 4 - riskSum * 120;
        if (battery < 15) score -= 60;
        if (health < 40) score -= 80;

        return Math.Max(1, score);
    }
}
