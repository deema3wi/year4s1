using System.Collections.ObjectModel;
using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Models;

public sealed class Mission : ObservableObject
{
    private GridPoint _base;
    private string _name = "Supply Run";
    private bool _delivered;
    private bool _returned;
    private bool _failed;

    public string Name
    {
        get => _name;
        set => Set(ref _name, value);
    }

    public GridPoint Base
    {
        get => _base;
        set => Set(ref _base, value);
    }

    public ObservableCollection<GridPoint> Targets { get; } = new();

    public bool Delivered
    {
        get => _delivered;
        set => Set(ref _delivered, value);
    }

    public bool Returned
    {
        get => _returned;
        set => Set(ref _returned, value);
    }

    public bool Failed
    {
        get => _failed;
        set => Set(ref _failed, value);
    }

    public void Reset(GridPoint @base)
    {
        Name = "Supply Run";
        Base = @base;
        Targets.Clear();
        Delivered = false;
        Returned = false;
        Failed = false;
    }
}
