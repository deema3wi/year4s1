using System;
using System.Collections.Generic;
using CMM.Ops.MissionControl.Algorithms;
using CMM.Ops.MissionControl.Models;

namespace CMM.Ops.MissionControl.Simulation;

public sealed class MissionSimulator
{
    private readonly AStarPathfinder _pathfinder = new();

    private List<GridPoint>? _path;
    private int _pathIndex;
    private double _moveAccumulator;

    public MissionPhase Phase { get; private set; } = MissionPhase.Idle;
    public bool IsRunning => Phase is MissionPhase.ToTarget or MissionPhase.ReturnToBase;

    public GridPoint? CurrentGoal { get; private set; }

    public event Action<string>? Log;
    public event Action? StateChanged;

    public void Reset()
    {
        _path = null;
        _pathIndex = 0;
        _moveAccumulator = 0;
        Phase = MissionPhase.Idle;
        CurrentGoal = null;
        StateChanged?.Invoke();
    }

    public bool Start(MapGrid map, Mission mission, DroneTwin drone, Inventory inv, double riskPenalty)
    {
        if (mission.Targets.Count == 0)
        {
            Log?.Invoke("No targets. Create a new mission.");
            return false;
        }

        var target = mission.Targets[0];
        var path = _pathfinder.FindPath(map, drone.Position, target, new AStarPathfinder.Options(riskPenalty));
        if (path is null)
        {
            Log?.Invoke("No path to target.");
            return false;
        }

        _path = path;
        _pathIndex = 0;
        _moveAccumulator = 0;
        Phase = MissionPhase.ToTarget;
        CurrentGoal = target;

        Log?.Invoke($"SIM start: to target {target} (riskPenalty={riskPenalty:0.0}).");
        StateChanged?.Invoke();
        return true;
    }

    public void Stop()
    {
        if (Phase is MissionPhase.ToTarget or MissionPhase.ReturnToBase)
        {
            Phase = MissionPhase.Idle;
            Log?.Invoke("SIM stopped.");
            StateChanged?.Invoke();
        }
    }

    public IReadOnlyList<GridPoint>? GetPath() => _path;

    public void Tick(double dtSeconds, MapGrid map, Mission mission, DroneTwin drone, Inventory inv, double riskPenalty, SimulationTuning tuning, Random rng)
    {
        if (!IsRunning || _path is null || CurrentGoal is null) return;

        // speed is cells per second; we move in discrete cell steps
        _moveAccumulator += dtSeconds * drone.SpeedCellsPerSecond;

        while (_moveAccumulator >= 1.0 && IsRunning)
        {
            _moveAccumulator -= 1.0;

            if (!StepOneCell(map, mission, drone, inv, riskPenalty, tuning, rng))
                break;
        }
    }

    private bool StepOneCell(MapGrid map, Mission mission, DroneTwin drone, Inventory inv, double riskPenalty, SimulationTuning tuning, Random rng)
    {
        if (_path is null || CurrentGoal is null) return false;

        // path index points to current cell in path
        if (_pathIndex >= _path.Count - 1)
        {
            OnReachedGoal(map, mission, drone, inv, riskPenalty, rng);
            return true;
        }

        var next = _path[_pathIndex + 1];

        // Obstacle appeared (editing during sim) -> replan
        if (map[next.X, next.Y].Type == CellType.Obstacle)
        {
            Log?.Invoke($"Obstacle on next cell {next}. Replanning...");
            if (!Replan(map, drone.Position, CurrentGoal.Value, riskPenalty))
                Fail(mission, "Replan failed: no route.");
            return false;
        }

        drone.Position = next;
        _pathIndex++;

        // Baseline consumption
        var mapRisk = map[next.X, next.Y].Risk;
        var tunedRisk = Math.Clamp(mapRisk * tuning.RiskMultiplier * drone.RiskSkillMultiplier, 0, 1);

        // Baseline battery + extra drain in risky zones (forces "safe routing" to matter)
        drone.DrainBattery(0.18 + drone.PayloadPct * 0.001 + tunedRisk * tuning.RiskBatteryDrain);
        inv.ConsumeFood(Phase == MissionPhase.ToTarget ? 1 : 0); // food for delivery phase
        inv.ConsumeAmmo(mapRisk > 0.6 ? 1 : 0);

        // More dangerous model:
        // - base chance + risk-based chance
        // - faster speed increases chance
        // - low health increases chance
        var speedFactor = 1.0 + Math.Max(0, drone.SpeedCellsPerSecond - 3.0) * 0.12;
        var healthFactor = 1.0 + (1.0 - drone.HealthPct / 100.0) * 0.80;
        var incidentProb = (tuning.BaseIncidentChance + tunedRisk * tuning.RiskIncidentFactor) * speedFactor * healthFactor;
        incidentProb = Math.Clamp(incidentProb, 0, 0.95);

        if (rng.NextDouble() < incidentProb)
        {
            var baseDmg = tuning.MinDamage + rng.NextDouble() * Math.Max(0.0, tuning.MaxDamage - tuning.MinDamage);
            var dmg = baseDmg * (0.60 + tunedRisk * 0.80) * tuning.DamageMultiplier;
            drone.ApplyDamage(dmg);
            drone.DrainBattery(tuning.ExtraBatteryOnIncident);
            Log?.Invoke($"INCIDENT at {next}: damage {dmg:0.0}% (p={incidentProb:0.00}, risk={tunedRisk:0.00}).");
        }

        // Storm event: high-risk zones can randomly drain extra battery.
        if (tunedRisk > 0.4 && rng.NextDouble() < tunedRisk * tuning.StormChance)
        {
            drone.DrainBattery(tuning.StormBatteryDrain);
            Log?.Invoke($"STORM at {next}: battery -{tuning.StormBatteryDrain:0.0}% (risk={tunedRisk:0.00}).");
        }

        if (drone.BatteryPct <= 0)
        {
            Fail(mission, "Battery depleted.");
            return false;
        }

        if (drone.HealthPct <= 0)
        {
            Fail(mission, "Drone destroyed.");
            return false;
        }

        // random dynamic obstacle event (low chance) to demonstrate avoidance
        if (rng.NextDouble() < tuning.DynamicObstacleChance && _pathIndex + 3 < _path.Count)
        {
            var block = _path[_pathIndex + 2];
            if (map[block.X, block.Y].Type == CellType.Empty && !(block.Equals(CurrentGoal.Value)))
            {
                map[block.X, block.Y].Type = CellType.Obstacle;
                Log?.Invoke($"DYNAMIC obstacle appeared at {block}.");
                if (!Replan(map, drone.Position, CurrentGoal.Value, riskPenalty))
                    Fail(mission, "Replan failed after dynamic obstacle.");
                return false;
            }
        }

        StateChanged?.Invoke();
        return true;
    }

    private void OnReachedGoal(MapGrid map, Mission mission, DroneTwin drone, Inventory inv, double riskPenalty, Random rng)
    {
        if (Phase == MissionPhase.ToTarget)
        {
            mission.Delivered = true;
            drone.DropPayload(40);
            Log?.Invoke($"DELIVERED at {CurrentGoal}. Returning to base...");
            Phase = MissionPhase.ReturnToBase;
            CurrentGoal = mission.Base;

            if (!Replan(map, drone.Position, CurrentGoal.Value, riskPenalty))
            {
                Fail(mission, "Cannot return: no route to base.");
                return;
            }

            StateChanged?.Invoke();
            return;
        }

        if (Phase == MissionPhase.ReturnToBase)
        {
            mission.Returned = true;

            RepairAtBase(drone, inv);

            Phase = MissionPhase.Completed;
            Log?.Invoke("MISSION COMPLETE.");
            StateChanged?.Invoke();
        }
    }

    private void RepairAtBase(DroneTwin drone, Inventory inv)
    {
        var missing = 100.0 - drone.HealthPct;
        if (missing <= 0.01) return;
        if (inv.Parts <= 0)
        {
            Log?.Invoke("WORKSHOP: no parts available for repair.");
            return;
        }

        // Workshop repair: each part restores more than field repair and is affected by repair efficiency.
        var healPerPart = 12.0 * drone.RepairEfficiency;
        if (healPerPart <= 0.01) healPerPart = 12.0;

        var needed = (int)Math.Ceiling(missing / healPerPart);
        var used = Math.Min(needed, inv.Parts);
        if (used <= 0) return;

        inv.TryConsumeParts(used);
        var healed = used * healPerPart;
        var before = drone.HealthPct;
        drone.HealthPct = Math.Min(100.0, drone.HealthPct + healed);

        Log?.Invoke($"WORKSHOP repair: used {used} parts, health {before:0}% -> {drone.HealthPct:0}%.");
    }

    private bool Replan(MapGrid map, GridPoint from, GridPoint to, double riskPenalty)
    {
        var path = _pathfinder.FindPath(map, from, to, new AStarPathfinder.Options(riskPenalty));
        if (path is null) return false;

        _path = path;
        _pathIndex = 0;
        return true;
    }

    private void Fail(Mission mission, string reason)
    {
        Phase = MissionPhase.Failed;
        mission.Failed = true;
        Log?.Invoke($"MISSION FAILED: {reason}");
        StateChanged?.Invoke();
    }
}
