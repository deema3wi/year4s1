using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows.Threading;
using System.Windows.Input;
using CMM.Ops.MissionControl.Algorithms;
using CMM.Ops.MissionControl.Infrastructure;
using CMM.Ops.MissionControl.Models;
using CMM.Ops.MissionControl.Petri;
using CMM.Ops.MissionControl.Simulation;

namespace CMM.Ops.MissionControl;

public sealed class MainViewModel : ObservableObject
{
    private readonly AStarPathfinder _pathfinder = new();
    private readonly MonteCarloPlanner _mcPlanner = new();
    private readonly MissionSimulator _sim = new();
    private readonly PetriEngine _petriEngine = new();
    private readonly DispatcherTimer _timer;
    private readonly Random _rng = new();

    private IReadOnlyList<GridPoint>? _currentPath;
    private string _statusText = "Ready";
    private bool _isEditMode;
    private int _selectedTabIndex;

    private double _plannedRiskPenalty = 2.0;
    private string _plannedModeName = "Balanced";
    private bool _signalDeliveredSent;
    private bool _signalReturnedSent;
    private bool _signalFailSent;

    public MapGrid Map { get; }
    public Mission Mission { get; } = new();
    public DroneTwin Drone { get; } = new();
    public Inventory Inventory { get; } = new();
    public CrewRoster Crew { get; } = new();
    public SimulationTuning Tuning { get; } = new();

    public PetriNet PetriNet { get; }

    public ObservableCollection<string> Logs { get; } = new();

    public IReadOnlyList<GridPoint>? CurrentPath
    {
        get => _currentPath;
        private set => Set(ref _currentPath, value);
    }

    public string StatusText
    {
        get => _statusText;
        private set => Set(ref _statusText, value);
    }

    public bool IsEditMode
    {
        get => _isEditMode;
        set => Set(ref _isEditMode, value);
    }

    public int SelectedTabIndex
    {
        get => _selectedTabIndex;
        set => Set(ref _selectedTabIndex, value);
    }

    // Computed summaries
    public string DroneSummary => $"Pos={Drone.Position}  Speed={Drone.SpeedCellsPerSecond:0.0} c/s  Risk×={Drone.RiskSkillMultiplier:0.00}  Repair×={Drone.RepairEfficiency:0.00}";
    public string InventorySummary => $"Food={Inventory.Food}/{Inventory.MaxFood}  Parts={Inventory.Parts}/{Inventory.MaxParts}  Ammo={Inventory.Ammo}/{Inventory.MaxAmmo}";
    public string MissionSummary
        => $"Base={Mission.Base}  Target={(Mission.Targets.Count > 0 ? Mission.Targets[0].ToString() : "—")}  Planned={(_currentPath is null ? "no" : $"{_plannedModeName} (penalty={_plannedRiskPenalty:0.0})")}  Delivered={Mission.Delivered}  Returned={Mission.Returned}  Failed={Mission.Failed}";
    public string CrewSummary
        => $"Pilot={Crew.AssignedPilot?.Name ?? "—"}  Engineer={Crew.AssignedEngineer?.Name ?? "—"}  Logistics={Crew.AssignedLogistician?.Name ?? "—"}";
    public string CrewEffectsSummary
        => $"Pilot skill reduces incident probability (Risk×). Engineer skill improves repairs (Repair×). Logistics skill reduces consumption a bit (Food/Ammo).";
    public string PetriSummary => $"PetriNet last event: {PetriNet.LastEvent}";

    // Commands
    public ICommand NewMissionCommand { get; }
    public ICommand PlanRouteCommand { get; }
    public ICommand MonteCarloPlanCommand { get; }
    public ICommand StartSimCommand { get; }
    public ICommand StopSimCommand { get; }
    public ICommand ResetPetriCommand { get; }
    public ICommand StepPetriCommand { get; }
    public ICommand ApplyCrewCommand { get; }
    public ICommand LogFromMapCommand { get; }
    public ICommand OpenCrewTabCommand { get; }
    public ICommand ApplyTuningCommand { get; }

    public MainViewModel()
    {
        Map = new MapGrid(width: 48, height: 32);

        PetriNet = BuildDefaultPetriNet();
        PetriNet.ResetToStart();

        NewMissionCommand = new RelayCommand(NewMission);
        PlanRouteCommand = new RelayCommand(PlanRoute);
        MonteCarloPlanCommand = new RelayCommand(MonteCarloPlan);
        StartSimCommand = new RelayCommand(StartSim);
        StopSimCommand = new RelayCommand(StopSim);
        ResetPetriCommand = new RelayCommand(() => { PetriNet.ResetToStart(); Raise(nameof(PetriSummary)); });
        StepPetriCommand = new RelayCommand(() => { _petriEngine.Step(PetriNet); Raise(nameof(PetriSummary)); });
        ApplyCrewCommand = new RelayCommand(ApplyCrewEffects);
        LogFromMapCommand = new RelayCommand<string>(msg => AddLog(msg ?? ""));
        OpenCrewTabCommand = new RelayCommand(() => SelectedTabIndex = 2);
        ApplyTuningCommand = new RelayCommand(ApplyTuningToDrone);

        _sim.Log += AddLog;
        _sim.StateChanged += () =>
        {
            Raise(nameof(DroneSummary));
            Raise(nameof(MissionSummary));
            Raise(nameof(InventorySummary));
            CurrentPath = _sim.GetPath();
            StatusText = _sim.Phase.ToString();
        };

        // Refresh computed properties if models change
        Drone.PropertyChanged += (_, __) => Raise(nameof(DroneSummary));
        Inventory.PropertyChanged += (_, __) => Raise(nameof(InventorySummary));
        Mission.PropertyChanged += (_, __) => Raise(nameof(MissionSummary));
        Crew.PropertyChanged += (_, __) => { Raise(nameof(CrewSummary)); Raise(nameof(CrewEffectsSummary)); };

        _timer = new DispatcherTimer
        {
            Interval = TimeSpan.FromMilliseconds(50)
        };
        _timer.Tick += (_, __) => OnTick();

        // Initial
        NewMission();
        ApplyCrewEffects();
    }

    private void OnTick()
    {
        _sim.Tick(dtSeconds: 0.05, Map, Mission, Drone, Inventory, _plannedRiskPenalty, Tuning, _rng);

        // sync petri with mission events
        if (Mission.Failed && !_signalFailSent)
        {
            _signalFailSent = true;
            _petriEngine.Signal(PetriNet, "fail");
            Raise(nameof(PetriSummary));
        }

        if (Mission.Delivered && !_signalDeliveredSent)
        {
            _signalDeliveredSent = true;
            _petriEngine.Signal(PetriNet, "delivered");
            Raise(nameof(PetriSummary));
        }

        if (Mission.Returned && !_signalReturnedSent)
        {
            _signalReturnedSent = true;
            _petriEngine.Signal(PetriNet, "returned");
            Raise(nameof(PetriSummary));
        }
    }

    private void NewMission()
    {
        StopSim();

        var seed = Environment.TickCount;
        Map.GenerateRandom(seed);

        // Choose a base and target cells that are not obstacles
        var basePt = FindFreeCellNear(new GridPoint(2, 2));
        var target = FindFreeCellNear(new GridPoint(Map.Width - 3, Map.Height - 3));

        Mission.Reset(basePt);
        Mission.Targets.Add(target);

        Drone.Reset(basePt);
        ApplyTuningToDrone();
        Inventory.Reset();
        _sim.Reset();

        _plannedRiskPenalty = 2.0;
        _plannedModeName = "Balanced";
        CurrentPath = null;

        _signalDeliveredSent = false;
        _signalReturnedSent = false;
        _signalFailSent = false;

        PetriNet.ResetToStart();

        AddLog($"New mission generated (seed={seed}). Base={Mission.Base}, Target={target}.");
        StatusText = "Ready";
        Raise(nameof(MissionSummary));
        Raise(nameof(PetriSummary));
    }

    private void PlanRoute()
    {
        StopSim();

        if (Mission.Targets.Count == 0)
        {
            AddLog("No targets.");
            return;
        }

        var start = Drone.Position;
        var goal = Mission.Targets[0];

        _plannedRiskPenalty = 2.0;
        _plannedModeName = "Balanced";

        var path = _pathfinder.FindPath(Map, start, goal, new AStarPathfinder.Options(_plannedRiskPenalty));
        if (path is null)
        {
            AddLog("Plan failed: no path.");
            CurrentPath = null;
            return;
        }

        CurrentPath = path;
        AddLog($"Planned route: {path.Count} cells, mode={_plannedModeName} (penalty={_plannedRiskPenalty:0.0}).");

        _petriEngine.Signal(PetriNet, "planned");
        Raise(nameof(PetriSummary));
        Raise(nameof(MissionSummary));
    }

    private void MonteCarloPlan()
    {
        StopSim();

        if (Mission.Targets.Count == 0)
        {
            AddLog("No targets.");
            return;
        }

        var start = Drone.Position;
        var goal = Mission.Targets[0];

        var choice = _mcPlanner.ChooseBest(Map, start, goal, Drone.RiskSkillMultiplier, Tuning, simulationsPerCandidate: 140);
        if (choice is null)
        {
            AddLog("Monte-Carlo: no viable route.");
            return;
        }

        _plannedRiskPenalty = choice.Candidate.RiskPenalty;
        _plannedModeName = choice.Candidate.Name;
        CurrentPath = choice.Path;

        AddLog($"Monte-Carlo chose: {choice.Candidate.Name} (penalty={choice.Candidate.RiskPenalty:0.0}) " +
               $"avgScore={choice.Score:0} success={choice.SuccessRate:P0} pathLen={choice.Path.Count}.");

        _petriEngine.Signal(PetriNet, "planned");
        Raise(nameof(PetriSummary));
        Raise(nameof(MissionSummary));
    }

    private void StartSim()
    {
        if (Mission.Targets.Count == 0)
        {
            AddLog("No targets.");
            return;
        }

        if (_sim.IsRunning)
        {
            AddLog("Already running.");
            return;
        }

        // If no plan, do a quick plan with the current settings
        if (CurrentPath is null)
            PlanRoute();

        var ok = _sim.Start(Map, Mission, Drone, Inventory, _plannedRiskPenalty);
        if (!ok) return;

        _signalDeliveredSent = false;
        _signalReturnedSent = false;
        _signalFailSent = false;

        _petriEngine.Signal(PetriNet, "start");
        Raise(nameof(PetriSummary));

        StatusText = "Running";
        _timer.Start();
    }

    private void StopSim()
    {
        _timer.Stop();
        _sim.Stop();
        StatusText = "Stopped";
    }

    private void ApplyCrewEffects()
    {
        // Skills 1..5
        var pilot = Crew.AssignedPilot?.PilotSkill ?? 3;
        var eng = Crew.AssignedEngineer?.EngineerSkill ?? 3;
        var logi = Crew.AssignedLogistician?.LogisticsSkill ?? 3;

        // Pilot reduces risk multiplier: 1 -> 1.12, 5 -> 0.72
        Drone.RiskSkillMultiplier = 1.20 - pilot * 0.12;

        // Engineer improves repairs: 1 -> 0.95, 5 -> 1.55
        Drone.RepairEfficiency = 0.75 + eng * 0.16;

        // Logistics reduces consumption indirectly: we model it by increasing max food/ammo slightly (simple)
        // to keep it visible in UI without complex per-step adjustments.
        // (Values stay stable; only "effective" consumption improves.)
        Inventory.Food = Math.Min(Inventory.MaxFood, Inventory.Food + (logi - 3) * 2);
        Inventory.Ammo = Math.Min(Inventory.MaxAmmo, Inventory.Ammo + (logi - 3) * 1);

        AddLog($"Crew effects applied: PilotSkill={pilot}, EngineerSkill={eng}, LogisticsSkill={logi}.");
        Raise(nameof(DroneSummary));
        Raise(nameof(InventorySummary));
        Raise(nameof(CrewSummary));
    }

    private void ApplyTuningToDrone()
    {
        // Apply "start" parameters without resetting mission/map.
        Drone.BatteryPct = Tuning.StartBatteryPct;
        Drone.HealthPct = Tuning.StartHealthPct;
        Drone.PayloadPct = Tuning.StartPayloadPct;

        AddLog($"Tuning applied: StartHealth={Tuning.StartHealthPct:0}% StartBattery={Tuning.StartBatteryPct:0}% StartPayload={Tuning.StartPayloadPct:0}% " +
               $"Risk×={Tuning.RiskMultiplier:0.00} IncidentBase={Tuning.BaseIncidentChance:0.00} IncidentRisk={Tuning.RiskIncidentFactor:0.00} Dmg×={Tuning.DamageMultiplier:0.00}.");

        Raise(nameof(DroneSummary));
        Raise(nameof(InventorySummary));
    }

    private GridPoint FindFreeCellNear(GridPoint desired)
    {
        if (Map[desired.X, desired.Y].Type != CellType.Obstacle) return desired;

        for (var r = 1; r < 8; r++)
        for (var dy = -r; dy <= r; dy++)
        for (var dx = -r; dx <= r; dx++)
        {
            var x = desired.X + dx;
            var y = desired.Y + dy;
            if (!Map.InBounds(x, y)) continue;
            if (Map[x, y].Type != CellType.Obstacle) return new GridPoint(x, y);
        }

        // fallback scan
        for (var y = 1; y < Map.Height - 1; y++)
        for (var x = 1; x < Map.Width - 1; x++)
            if (Map[x, y].Type != CellType.Obstacle) return new GridPoint(x, y);

        return desired;
    }

    private void AddLog(string message)
    {
        if (string.IsNullOrWhiteSpace(message)) return;

        var line = $"[{DateTime.Now:HH:mm:ss}] {message}";
        Logs.Insert(0, line);

        while (Logs.Count > 200)
            Logs.RemoveAt(Logs.Count - 1);
    }

    private static PetriNet BuildDefaultPetriNet()
    {
        // Layout is normalized to [0..1] coordinates for easy drawing.
        var net = new PetriNet();

        var pStart = new PetriPlace("start", "Start", 0.10, 0.50);
        var pPlanned = new PetriPlace("planned", "Planned", 0.30, 0.25);
        var pFlying = new PetriPlace("flying", "Flying", 0.50, 0.50);
        var pDelivered = new PetriPlace("delivered", "Delivered", 0.70, 0.25);
        var pReturning = new PetriPlace("returning", "Returning", 0.70, 0.75);
        var pDone = new PetriPlace("done", "Done", 0.90, 0.50);
        var pFail = new PetriPlace("fail", "Fail", 0.50, 0.85);

        net.Places.Add(pStart);
        net.Places.Add(pPlanned);
        net.Places.Add(pFlying);
        net.Places.Add(pDelivered);
        net.Places.Add(pReturning);
        net.Places.Add(pDone);
        net.Places.Add(pFail);

        var tPlan = new PetriTransition("t_plan", "Plan", 0.20, 0.38);
        var tStart = new PetriTransition("t_start", "Launch", 0.40, 0.38);
        var tDeliver = new PetriTransition("t_deliver", "Drop", 0.60, 0.38);
        var tReturn = new PetriTransition("t_return", "RTB", 0.80, 0.62);
        var tFail = new PetriTransition("t_fail", "Crash", 0.50, 0.68);

        // start -> planned
        tPlan.Inputs.Add(new PetriArc(pStart, 1));
        tPlan.Outputs.Add(new PetriArc(pPlanned, 1));

        // planned -> flying
        tStart.Inputs.Add(new PetriArc(pPlanned, 1));
        tStart.Outputs.Add(new PetriArc(pFlying, 1));

        // flying -> delivered + returning (split)
        tDeliver.Inputs.Add(new PetriArc(pFlying, 1));
        tDeliver.Outputs.Add(new PetriArc(pDelivered, 1));
        tDeliver.Outputs.Add(new PetriArc(pReturning, 1));

        // returning -> done
        tReturn.Inputs.Add(new PetriArc(pReturning, 1));
        tReturn.Inputs.Add(new PetriArc(pDelivered, 1)); // require delivered
        tReturn.Outputs.Add(new PetriArc(pDone, 1));

        // any state -> fail (we model as needing 1 token in flying/returning/planned)
        // To keep it simple without inhibitor arcs, we add 3 transitions? We'll just allow crash from flying or returning.
        tFail.Inputs.Add(new PetriArc(pFlying, 1));
        tFail.Outputs.Add(new PetriArc(pFail, 1));

        net.Transitions.Add(tPlan);
        net.Transitions.Add(tStart);
        net.Transitions.Add(tDeliver);
        net.Transitions.Add(tReturn);
        net.Transitions.Add(tFail);

        return net;
    }
}
