namespace ServerTwin.Services;

public partial class ServerSecurityMonitoringService(ServerDigitalTwin twin,
	IAIModel<ServerSecurityInput, ServerSecurityOutput> securityModel)
{
	private readonly ServerDigitalTwin _twin = twin;
	private readonly IAIModel<ServerSecurityInput, ServerSecurityOutput> _securityModel = securityModel;
}

public partial class ServerSecurityMonitoringService
{
	public ServerSecurityOutput Analyze()
	{
		var s = _twin.State;
		ServerSecurityInput input = new(
			s.CpuUsage,
			s.NetInMbps,
			s.NetOutMbps,
			s.ActiveConnections,
			s.ErrorRatePerMinute
		);

		var result = _securityModel.Predict(input);

		if (result.IsAnomaly)
		{
			Console.WriteLine($"[SECURITY] Anomaly={result.AnomalyScore:F2}{Environment.NewLine}" +
				$"DDos={result.IsSuspectedDdos}, Bruteforce={result.IsSuspectedBruteForce}");
		}

		return result;
	}
}