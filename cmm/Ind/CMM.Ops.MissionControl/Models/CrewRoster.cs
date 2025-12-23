using System.Collections.ObjectModel;
using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Models;

public sealed class CrewRoster : ObservableObject
{
    private CrewMember? _assignedPilot;
    private CrewMember? _assignedEngineer;
    private CrewMember? _assignedLogistician;

    public ObservableCollection<CrewMember> People { get; } = new();

    public CrewMember? AssignedPilot
    {
        get => _assignedPilot;
        set => Set(ref _assignedPilot, value);
    }

    public CrewMember? AssignedEngineer
    {
        get => _assignedEngineer;
        set => Set(ref _assignedEngineer, value);
    }

    public CrewMember? AssignedLogistician
    {
        get => _assignedLogistician;
        set => Set(ref _assignedLogistician, value);
    }

    public CrewRoster()
    {
        People.Add(new CrewMember { Name = "Danylo", PilotSkill = 5, EngineerSkill = 2, LogisticsSkill = 2 });
        People.Add(new CrewMember { Name = "Iryna",  PilotSkill = 3, EngineerSkill = 4, LogisticsSkill = 3 });
        People.Add(new CrewMember { Name = "Oleh",   PilotSkill = 2, EngineerSkill = 5, LogisticsSkill = 2 });
        People.Add(new CrewMember { Name = "Sofiia", PilotSkill = 3, EngineerSkill = 2, LogisticsSkill = 5 });
        People.Add(new CrewMember { Name = "Taras",  PilotSkill = 4, EngineerSkill = 3, LogisticsSkill = 3 });

        AssignedPilot = People[0];
        AssignedEngineer = People[2];
        AssignedLogistician = People[3];
    }
}
