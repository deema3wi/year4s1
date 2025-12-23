namespace ServerTwin.Models;

public sealed record AutoScalingResult(
	DateTime TimeStampUtc,
	float Risk,
	int ModelDelta,
	int AppliedDelta,
	int ReplicasBefore,
	int ReplicasAfter,
	string Reason
);
