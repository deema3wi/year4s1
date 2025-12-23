using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Petri;

public sealed class PetriPlace : ObservableObject
{
    private int _tokens;

    public PetriPlace(string id, string name, double x, double y)
    {
        Id = id;
        Name = name;
        X = x;
        Y = y;
    }

    public string Id { get; }
    public string Name { get; }
    public double X { get; }
    public double Y { get; }

    public int Tokens
    {
        get => _tokens;
        set => Set(ref _tokens, value < 0 ? 0 : value);
    }
}
