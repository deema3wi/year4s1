namespace ServerTwin.Models;

public record CapacityPlanningForecastPoint(
	DateTime TimeStamp,
	double ExpectedCpuUsage,
	double ExpectedRamUsage,
	int ExpectedActiveConnections
);