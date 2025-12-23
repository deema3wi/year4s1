using System;

namespace CMM.Ops.MissionControl.Petri;

public sealed class PetriEngine
{
    public bool Step(PetriNet net)
    {
        foreach (var t in net.Transitions)
        {
            if (t.IsEnabled())
            {
                t.Fire();
                net.LastEvent = $"Fired: {t.Name}";
                return true;
            }
        }

        net.LastEvent = "No enabled transitions";
        return false;
    }

    public void Signal(PetriNet net, string signal)
    {
        net.LastEvent = $"Signal: {signal}";

        if (signal == "fail")
        {
            foreach (var p in net.Places) p.Tokens = 0;
            net.FindPlace("fail")!.Tokens = 1;
            net.LastEvent = "Signal applied: Fail";
            return;
        }

        PetriTransition? t = signal switch
        {
            "planned" => net.FindTransition("t_plan"),
            "start" => net.FindTransition("t_start"),
            "delivered" => net.FindTransition("t_deliver"),
            "returned" => net.FindTransition("t_return"),
            _ => null
        };

        if (t is null) return;

        if (t.Fire())
            net.LastEvent = $"Signal fired: {t.Name}";
        else
            net.LastEvent = $"Signal ignored (transition disabled): {t.Name}";
    }
}
