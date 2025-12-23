using System.Collections.ObjectModel;
using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Petri;

public sealed class PetriNet : ObservableObject
{
    private string _lastEvent = "—";

    public ObservableCollection<PetriPlace> Places { get; } = new();
    public ObservableCollection<PetriTransition> Transitions { get; } = new();

    public string LastEvent
    {
        get => _lastEvent;
        set => Set(ref _lastEvent, value);
    }

    public void ResetToStart()
    {
        foreach (var p in Places) p.Tokens = 0;
        var start = FindPlace("start");
        if (start is not null) start.Tokens = 1;
        LastEvent = "Reset";
    }

    public PetriPlace? FindPlace(string id)
    {
        foreach (var p in Places)
            if (p.Id == id) return p;
        return null;
    }

    public PetriTransition? FindTransition(string id)
    {
        foreach (var t in Transitions)
            if (t.Id == id) return t;
        return null;
    }
}
