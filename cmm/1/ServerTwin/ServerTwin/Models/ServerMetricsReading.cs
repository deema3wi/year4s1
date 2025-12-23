namespace ServerTwin.Models;

public record ServerMetricsReading(
	DateTime TimeStamp,
	double CpuUsagePercentage,
	double RamUsagePercentage,
	double DiskUsagePercentage,
	double DiskIOPs,
	double NetInMbps,
	double NetOutMbps,
	double TemperatureC,
	int ActiveConnections,
	double ErrorRatePerMinute
);