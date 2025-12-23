namespace ServerTwin.Models;

public record ServerSecurityOutput(
	float AnomalyScore,
	bool IsAnomaly,
	bool IsSuspectedDdos,
	bool IsSuspectedBruteForce
);