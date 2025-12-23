using System;
using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Models;

/// <summary>
/// Gameplay/simulation tuning knobs shown in the UI.
/// </summary>
public sealed class SimulationTuning : ObservableObject
{
    private double _startBatteryPct = 90;
    private double _startHealthPct = 65;
    private double _startPayloadPct = 60;

    private double _riskMultiplier = 1.15;
    private double _baseIncidentChance = 0.02;
    private double _riskIncidentFactor = 0.28;
    private double _damageMultiplier = 1.25;

    private double _minDamage = 8;
    private double _maxDamage = 18;

    private double _riskBatteryDrain = 0.55;
    private double _extraBatteryOnIncident = 0.8;

    private double _dynamicObstacleChance = 0.03;

    private double _stormChance = 0.02;
    private double _stormBatteryDrain = 1.4;

    public double StartBatteryPct
    {
        get => _startBatteryPct;
        set => Set(ref _startBatteryPct, Math.Clamp(value, 20, 100));
    }

    public double StartHealthPct
    {
        get => _startHealthPct;
        set => Set(ref _startHealthPct, Math.Clamp(value, 20, 100));
    }

    public double StartPayloadPct
    {
        get => _startPayloadPct;
        set => Set(ref _startPayloadPct, Math.Clamp(value, 0, 100));
    }

    /// <summary>Risk value from the map is multiplied by this (higher = more dangerous).</summary>
    public double RiskMultiplier
    {
        get => _riskMultiplier;
        set => Set(ref _riskMultiplier, Math.Clamp(value, 0.5, 2.5));
    }

    /// <summary>Flat incident chance per step (0..1).</summary>
    public double BaseIncidentChance
    {
        get => _baseIncidentChance;
        set => Set(ref _baseIncidentChance, Math.Clamp(value, 0.0, 0.25));
    }

    /// <summary>Additional incident chance at risk=1 (0..1).</summary>
    public double RiskIncidentFactor
    {
        get => _riskIncidentFactor;
        set => Set(ref _riskIncidentFactor, Math.Clamp(value, 0.0, 0.8));
    }

    public double DamageMultiplier
    {
        get => _damageMultiplier;
        set => Set(ref _damageMultiplier, Math.Clamp(value, 0.5, 3.0));
    }

    public double MinDamage
    {
        get => _minDamage;
        set => Set(ref _minDamage, Math.Clamp(value, 1, 40));
    }

    public double MaxDamage
    {
        get => _maxDamage;
        set => Set(ref _maxDamage, Math.Clamp(value, 1, 60));
    }

    /// <summary>Extra battery drain per step at risk=1.</summary>
    public double RiskBatteryDrain
    {
        get => _riskBatteryDrain;
        set => Set(ref _riskBatteryDrain, Math.Clamp(value, 0.0, 3.0));
    }

    public double ExtraBatteryOnIncident
    {
        get => _extraBatteryOnIncident;
        set => Set(ref _extraBatteryOnIncident, Math.Clamp(value, 0.0, 5.0));
    }

    /// <summary>Per-step chance to spawn a dynamic obstacle somewhere ahead (0..1).</summary>
    public double DynamicObstacleChance
    {
        get => _dynamicObstacleChance;
        set => Set(ref _dynamicObstacleChance, Math.Clamp(value, 0.0, 0.25));
    }

    /// <summary>Per-step chance of a "storm" event at risk=1 (0..1).</summary>
    public double StormChance
    {
        get => _stormChance;
        set => Set(ref _stormChance, Math.Clamp(value, 0.0, 0.25));
    }

    public double StormBatteryDrain
    {
        get => _stormBatteryDrain;
        set => Set(ref _stormBatteryDrain, Math.Clamp(value, 0.0, 6.0));
    }
}
