namespace ServerTwin.Services;

public sealed class ClusterSecurityMonitoringService
{
	private readonly ServerClusterTwin _cluster;
	private readonly IAIModel<ServerSecurityInput, ServerSecurityOutput> _securityModel;

	public ClusterSecurityMonitoringService(
		ServerClusterTwin cluster,
		IAIModel<ServerSecurityInput, ServerSecurityOutput> securityModel)
	{
		_cluster = cluster;
		_securityModel = securityModel;
	}

	public ServerSecurityOutput Analyze()
	{
		var input = _cluster.GetSecurityInputPerReplica();
		var result = _securityModel.Predict(input);

		if (result.IsAnomaly)
			_cluster.Events.Add(new ClusterEvent(DateTime.UtcNow,
				$"[SECURITY] anomalyScore={result.AnomalyScore:F2}, DDoS={result.IsSuspectedDdos}, Bruteforce={result.IsSuspectedBruteForce}"));

		return result;
	}
}
