namespace CMM.Ops.MissionControl.Models;

public readonly record struct GridPoint(int X, int Y)
{
    public override string ToString() => $"({X},{Y})";
}
