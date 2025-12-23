namespace ServerTwin.Models;

public sealed record ClusterMetrics(
	DateTime TimeStampUtc,
	int Replicas,
	double AvgCpu,
	double AvgRam,
	double AvgDisk,
	double AvgDiskIOPs,
	double AvgNetInMbps,
	double AvgNetOutMbps,
	int AvgConnections,
	double AvgErrorRatePerMinute,
	double AvgTemperatureC
);
