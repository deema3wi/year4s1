namespace ServerTwin.Helpers;

public static class GenerateReading
{
	public static ServerMetricsReading GenerateSyntheticMetrics(Random rnd, DateTime timestamp)
	{
		double secondsOfDay = (timestamp - DateTime.UtcNow.Date).TotalSeconds;

		double baseCpu = 40 + 30 * Math.Sin(secondsOfDay / 120.0);
		double noiseCpu = rnd.NextDouble() * 10 - 5;
		double cpu = ClampHelper.Clamp(baseCpu + noiseCpu, 0, 100);

		double baseRam = 50 + cpu * 0.3;
		double noiseRam = rnd.NextDouble() * 10 - 5;
		double ram = ClampHelper.Clamp(baseRam + noiseRam, 0, 100);

		double diskUsage = ClampHelper.Clamp(60 + 10 * Math.Sin(secondsOfDay / 300.0), 0, 100);
		double diskIops = 50 + cpu * 5 + rnd.Next(0, 200);

		double netIn = Math.Max(0, cpu * 0.8 + rnd.NextDouble() * 50);
		double netOut = Math.Max(0, cpu * 0.5 + rnd.NextDouble() * 30);

		int activeConn = (int)(cpu * 20 + rnd.Next(-50, 51));
		if (activeConn < 0) activeConn = 0;

		double errorRate = cpu > 90
			? rnd.Next(20, 120)
			: rnd.Next(0, 20);

		double temperature = 30 + cpu * 0.3;

		return new ServerMetricsReading(
			TimeStamp: timestamp,
			CpuUsagePercentage: cpu,
			RamUsagePercentage: ram,
			DiskUsagePercentage: diskUsage,
			DiskIOPs: diskIops,
			NetInMbps: netIn,
			NetOutMbps: netOut,
			TemperatureC: temperature,
			ActiveConnections: activeConn,
			ErrorRatePerMinute: errorRate
		);
	}
}