using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Models;

public sealed class MapCell : ObservableObject
{
    private CellType _type;
    private double _risk; // 0..1

    public MapCell(CellType type, double risk)
    {
        _type = type;
        _risk = risk;
    }

    public CellType Type
    {
        get => _type;
        set => Set(ref _type, value);
    }

    public double Risk
    {
        get => _risk;
        set
        {
            var v = value;
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            Set(ref _risk, v);
        }
    }
}
