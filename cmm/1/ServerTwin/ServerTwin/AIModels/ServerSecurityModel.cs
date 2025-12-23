namespace ServerTwin.AIModels;

public class ServerSecurityModel : IAIModel<ServerSecurityInput, ServerSecurityOutput>
{
	public ServerSecurityOutput Predict(ServerSecurityInput input)
	{
		double score = 0.0d;
		double netTotal = input.NetInMbps + input.NetOutMbps;
		score += netTotal switch
		{
			> 1000.0d => 0.4d,
			> 300.0d => 0.25d,
			> 100.0d => 0.1d,
			_ => 0d,
		};

		score += input.ActiveConnections switch
		{
			> 5000 => 0.3d,
			> 2000 => 0.2d,
			> 800 => 0.1d,
			_ => 0d,
		};

		score += input.ErrorRatePerMinute switch
		{
			> 100.0d => 0.4d,
			> 40.0d => 0.25d,
			> 10.0d => 0.1d,
			_ => 0d,
		};

		score += input.CpuUsage switch
		{
			> 90.0d => 0.2d,
			> 70.0d => 0.1d,
			_ => 0d,
		};

		score = ClampHelper.Clamp01(score);

		bool isAnomaly = score >= 0.5d;
		bool isSuspectedDdos = netTotal > 300.0d && input.ErrorRatePerMinute > 10 && input.ActiveConnections > 2000;
		bool isSuspectedBruteForce = input.ErrorRatePerMinute > 40.0d && input.ActiveConnections > 300 && netTotal < 300;

		return new((float)score, isAnomaly, isSuspectedDdos, isSuspectedBruteForce);
	}
}