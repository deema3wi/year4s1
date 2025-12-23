namespace ServerTwin.Helpers;

public static class WorkloadGenerator
{
	public static WorkloadReading Generate(Random rnd, DateTime timestampUtc, SimulationPreset preset)
	{
		double secondsOfDay = (timestampUtc - timestampUtc.Date).TotalSeconds;
		double wave = 0.5 + 0.5 * Math.Sin(secondsOfDay / 180.0);

		var noiseCpu = (rnd.NextDouble() * 2 - 1) * (preset == SimulationPreset.Dynamic ? 18 : 8);
		var baseCpu = 35 + 40 * wave + noiseCpu;

		double spikeMult = 1.0;
		string? tag = null;

		if (preset == SimulationPreset.Dynamic)
		{
			if (rnd.NextDouble() < 0.08)
			{
				spikeMult = 1.8 + rnd.NextDouble() * 2.5;
				tag = "spike";
			}

			if (rnd.NextDouble() < 0.03)
			{
				spikeMult = Math.Max(spikeMult, 2.5 + rnd.NextDouble() * 3.0);
				tag = "burst";
			}
		}

		var cpuDemand = Math.Max(0, baseCpu) * spikeMult;
		var ramDemand = Math.Max(0, 45 + cpuDemand * 0.35 + (rnd.NextDouble() * 2 - 1) * (preset == SimulationPreset.Dynamic ? 10 : 6));

		var netIn = Math.Max(0, cpuDemand * 1.2 + rnd.NextDouble() * (preset == SimulationPreset.Dynamic ? 220 : 80));
		var netOut = Math.Max(0, cpuDemand * 0.8 + rnd.NextDouble() * (preset == SimulationPreset.Dynamic ? 140 : 50));
		var conns = (int)Math.Max(0, cpuDemand * 55 + rnd.Next(-150, 151));

		double err = cpuDemand > 95
			? rnd.Next(25, preset == SimulationPreset.Dynamic ? 180 : 120)
			: rnd.Next(0, preset == SimulationPreset.Dynamic ? 25 : 15);

		var diskBase = ClampHelper.Clamp(62 + 8 * Math.Sin(secondsOfDay / 600.0) + (rnd.NextDouble() * 2 - 1) * 1.5, 0, 100);
		var diskIops = 40 + cpuDemand * 6 + rnd.Next(0, preset == SimulationPreset.Dynamic ? 400 : 220);

		return new WorkloadReading(
			TimeStamp: timestampUtc,
			CpuDemand: cpuDemand,
			RamDemand: ramDemand,
			NetInDemandMbps: netIn,
			NetOutDemandMbps: netOut,
			ConnectionDemand: conns,
			ErrorRateDemandPerMinute: err,
			DiskBaseUsagePercentage: diskBase,
			DiskIopsDemand: diskIops,
			Tag: tag
		);
	}
}
