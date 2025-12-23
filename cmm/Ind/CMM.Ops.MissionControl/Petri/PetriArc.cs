namespace CMM.Ops.MissionControl.Petri;

public sealed class PetriArc
{
    public PetriArc(PetriPlace place, int weight)
    {
        Place = place;
        Weight = weight;
    }

    public PetriPlace Place { get; }
    public int Weight { get; }
}
