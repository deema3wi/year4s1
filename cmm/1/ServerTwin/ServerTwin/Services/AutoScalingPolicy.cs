namespace ServerTwin.Services;

public sealed record AutoScalingPolicy(
	int MinReplicas,
	int MaxReplicas,
	TimeSpan Cooldown,
	TimeSpan LowRiskHoldTime,
	int MaxScaleOutPerAction = 2,
	int MaxScaleInPerAction = 1,
	double LowRiskThreshold = 0.18,
	double HighRiskThreshold = 0.70
);
