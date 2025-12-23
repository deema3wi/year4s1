namespace ServerTwin.Models;

public record ServerLoadPredictionOutput(
	[property: Range(0, 1)]
	float OverloadRisk,
	int RecommendedReplicaDelta
);