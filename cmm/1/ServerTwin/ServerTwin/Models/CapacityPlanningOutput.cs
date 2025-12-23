namespace ServerTwin.Models;

public record CapacityPlanningOutput(
	IReadOnlyList<CapacityPlanningForecastPoint> Forecasts,
	string Recommendation
);