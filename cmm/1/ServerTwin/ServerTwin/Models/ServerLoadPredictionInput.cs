namespace ServerTwin.Models;

public record ServerLoadPredictionInput(
	double CurrentCpuUsage,
	double CurrentMemoryUsage,
	double NetInMbps,
	double NetOutMbps,
	int ActiveConnections
);