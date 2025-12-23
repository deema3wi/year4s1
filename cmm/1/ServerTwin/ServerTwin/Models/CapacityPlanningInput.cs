namespace ServerTwin.Models;

public record CapacityPlanningInput(
	string ServerId,
	ServerRole Role,
	IReadOnlyCollection<ServerMetricsReading> History
);