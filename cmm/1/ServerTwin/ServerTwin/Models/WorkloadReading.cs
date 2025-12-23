namespace ServerTwin.Models;

public sealed record WorkloadReading(
	DateTime TimeStamp,
	double CpuDemand,
	double RamDemand,
	double NetInDemandMbps,
	double NetOutDemandMbps,
	int ConnectionDemand,
	double ErrorRateDemandPerMinute,
	double DiskBaseUsagePercentage,
	double DiskIopsDemand,
	string? Tag = null
);
