namespace CMM.Ops.MissionControl.Models;

public sealed class CrewMember
{
    public required string Name { get; init; }
    public int PilotSkill { get; init; }
    public int EngineerSkill { get; init; }
    public int LogisticsSkill { get; init; }

    public override string ToString() => Name;
}
