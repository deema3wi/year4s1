namespace ServerTwin.AIModels;

public class CapacityPlanningModel : IAIModel<CapacityPlanningInput, CapacityPlanningOutput>
{
	public CapacityPlanningOutput Predict(CapacityPlanningInput input)
	{
		var history = input.History.ToList();
		if (history.Count == 0)
		{
			var now = DateTime.UtcNow;
			var forecast = Enumerable.Range(1, 24)
				.Select(h => new CapacityPlanningForecastPoint(now.AddHours(h),	50,	50,	200))
				.ToList();

			return new CapacityPlanningOutput(
				forecast,
				Recommendation: Recommendations.NotEnoughHistory
			);
		}

		var avgCpu = history.Average(h => h.CpuUsagePercentage);
		var avgRam = history.Average(h => h.RamUsagePercentage);
		var avgConn = history.Average(h => h.ActiveConnections);

		var peakCpu = history.Max(h => h.CpuUsagePercentage);
		var peakRam = history.Max(h => h.RamUsagePercentage);

		double forecastCpu = ClampHelper.Clamp01(avgCpu * 1.1) * 100.0;
		double forecastRam = ClampHelper.Clamp01(avgRam * 1.1) * 100.0;
		int forecastConn = (int)(avgConn * 1.1);

		var nowBase = DateTime.UtcNow;
		var points = Enumerable.Range(1, 24)
			.Select(h => new CapacityPlanningForecastPoint(
				TimeStamp: nowBase.AddHours(h),
				ExpectedCpuUsage: forecastCpu,
				ExpectedRamUsage: forecastRam,
				ExpectedActiveConnections: forecastConn))
			.ToList();

		string recommendation = BuildRecommendation(
			input.Role, avgCpu, avgRam, avgConn, peakCpu, peakRam);

		return new(points, recommendation);
	}

	private static string BuildRecommendation(
		ServerRole role,
		double avgCpu, double avgRam, double avgConn,
		double peakCpu, double peakRam)
	{
		if (peakCpu > 85 || avgCpu > 70)
		{
			return role switch
			{
				ServerRole.DB => Recommendations.CPUOverload,
				_ => Recommendations.CPUNearLimit
			};
		}

		if (peakRam > 85 || avgRam > 75)
		{
			return role switch
			{
				ServerRole.DB => Recommendations.RAMOverload,
				_ => Recommendations.RAMNearLimit
			};
		}

		bool lowUtil = avgCpu < 40 && avgRam < 40 && avgConn < 200;
		return lowUtil switch
		{
			true => Recommendations.ServerLowUtilization,
			false => Recommendations.Ok
		};
	}

}