namespace ServerTwin.Models;

public record ServerSecurityInput(
	double CpuUsage,
	double NetInMbps,
	double NetOutMbps,
	int ActiveConnections,
	double ErrorRatePerMinute
);