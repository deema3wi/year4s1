using System;
using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Models;

public sealed class DroneTwin : ObservableObject
{
    private GridPoint _position;
    private double _battery; // 0..100
    private double _health;  // 0..100
    private double _payload; // 0..100 (relative)
    private double _speedCellsPerSecond;
    private double _riskSkillMultiplier; // crew effect
    private double _repairEfficiency;    // crew effect

    public GridPoint Position
    {
        get => _position;
        set => Set(ref _position, value);
    }

    public double BatteryPct
    {
        get => _battery;
        set => Set(ref _battery, Math.Clamp(value, 0, 100));
    }

    public double HealthPct
    {
        get => _health;
        set => Set(ref _health, Math.Clamp(value, 0, 100));
    }

    public double PayloadPct
    {
        get => _payload;
        set => Set(ref _payload, Math.Clamp(value, 0, 100));
    }

    public double SpeedCellsPerSecond
    {
        get => _speedCellsPerSecond;
        set => Set(ref _speedCellsPerSecond, Math.Clamp(value, 0.5, 8));
    }

    /// <summary>Lower is better (risk is multiplied by this).</summary>
    public double RiskSkillMultiplier
    {
        get => _riskSkillMultiplier;
        set => Set(ref _riskSkillMultiplier, Math.Clamp(value, 0.65, 1.2));
    }

    /// <summary>Higher is better (repair cost is divided by this).</summary>
    public double RepairEfficiency
    {
        get => _repairEfficiency;
        set => Set(ref _repairEfficiency, Math.Clamp(value, 0.8, 1.6));
    }

    public DroneTwin()
    {
        Reset(new GridPoint(2, 2));
    }

    public void Reset(GridPoint start)
    {
        Position = start;
        BatteryPct = 100;
        HealthPct = 100;
        PayloadPct = 50;
        SpeedCellsPerSecond = 3.0;

        RiskSkillMultiplier = 1.0;
        RepairEfficiency = 1.0;
    }

    public void ApplyDamage(double amount)
    {
        if (amount <= 0) return;
        HealthPct -= amount;

        // Damage slows the drone a bit.
        SpeedCellsPerSecond = Math.Max(1.0, SpeedCellsPerSecond - amount * 0.02);
    }

    public void DrainBattery(double amount)
    {
        if (amount <= 0) return;
        BatteryPct -= amount;
    }

    public void DropPayload(double amount)
    {
        if (amount <= 0) return;
        PayloadPct -= amount;
    }
}
