using System;
using System.Collections.Generic;

namespace CMM.Ops.MissionControl.Petri;

public sealed class PetriTransition
{
    public PetriTransition(string id, string name, double x, double y)
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

    public List<PetriArc> Inputs { get; } = new();
    public List<PetriArc> Outputs { get; } = new();

    public bool IsEnabled()
    {
        foreach (var a in Inputs)
            if (a.Place.Tokens < a.Weight) return false;
        return true;
    }

    public bool Fire()
    {
        if (!IsEnabled()) return false;

        foreach (var a in Inputs)
            a.Place.Tokens -= a.Weight;

        foreach (var a in Outputs)
            a.Place.Tokens += a.Weight;

        return true;
    }
}
