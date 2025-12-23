namespace ServerTwin.Helpers;

public static class PrintHelper
{
	public static void PrintCluster(
		ServerClusterTwin cluster,
		WorkloadReading workload,
		AutoScalingResult scale,
		ServerSecurityOutput sec,
		SimulationPreset workloadPreset,
		VisualizationPreset visualPreset)
	{
		Console.Clear();

		var m = cluster.LastClusterMetrics;
		Console.WriteLine($"Time: {m.TimeStampUtc:HH:mm:ss} UTC");
		Console.WriteLine($"Cluster Role: {cluster.Role}   Replicas: {m.Replicas}   Workload: {workloadPreset}   Visual: {visualPreset}");
		Console.WriteLine("Keys: [R]ealistic  [D]ynamic  [V]isual toggle  [Q]uit");
		Console.WriteLine();

		if (visualPreset == VisualizationPreset.Dynamic)
		{
			Console.WriteLine($"CPU: {Bar(m.AvgCpu)}   RAM: {Bar(m.AvgRam)}   Disk: {Bar(m.AvgDisk)}");
			Console.WriteLine($"Net In: {m.AvgNetInMbps,6:F1} Mbps   Net Out: {m.AvgNetOutMbps,6:F1} Mbps   Conn(avg): {m.AvgConnections}");
			Console.WriteLine($"IOPs(avg): {m.AvgDiskIOPs,6:F0}   Err/min(avg): {m.AvgErrorRatePerMinute,6:F1}   Temp(avg): {m.AvgTemperatureC,5:F1} °C");
		}
		else
		{
			Console.WriteLine($"CPU: {m.AvgCpu:F1}%   RAM: {m.AvgRam:F1}%   Disk: {m.AvgDisk:F1}%");
			Console.WriteLine($"Net: In {m.AvgNetInMbps:F1} Mbps  Out {m.AvgNetOutMbps:F1} Mbps  Conn(avg): {m.AvgConnections}");
			Console.WriteLine($"IOPs(avg): {m.AvgDiskIOPs:F0}  Err/min(avg): {m.AvgErrorRatePerMinute:F1}  Temp(avg): {m.AvgTemperatureC:F1} °C");
		}

		Console.WriteLine();
		Console.WriteLine($"[AUTOSCALE] risk={scale.Risk:F2}, modelDelta={scale.ModelDelta}, appliedDelta={scale.AppliedDelta}, replicas {scale.ReplicasBefore} -> {scale.ReplicasAfter} ({scale.Reason})");
		Console.WriteLine($"[SECURITY]  anomalyScore={sec.AnomalyScore:F2}, isAnomaly={sec.IsAnomaly}, DDoS={sec.IsSuspectedDdos}, Bruteforce={sec.IsSuspectedBruteForce}");
		Console.WriteLine($"[WORKLOAD]  cpuDemand={workload.CpuDemand:F0}, connDemand={workload.ConnectionDemand}, tag={(workload.Tag ?? "-")}");

		Console.WriteLine();
		Console.WriteLine("Replicas:");

		var replicas = cluster.Replicas;
		var show = visualPreset == VisualizationPreset.Dynamic ? Math.Min(12, replicas.Count) : Math.Min(6, replicas.Count);

		for (var i = 0; i < show; i++)
		{
			var r = replicas[i];
			var s = r.State;
			if (visualPreset == VisualizationPreset.Dynamic)
			{
				Console.WriteLine($"  {s.ServerId,-8} {s.Status,-8} CPU {Bar(s.CpuUsage)}  RAM {Bar(s.RamUsage)}  Conn {s.ActiveConnections,5}  Err {s.ErrorRatePerMinute,5:F1}");
			}
			else
			{
				Console.WriteLine($"  {s.ServerId,-8} {s.Status,-8} CPU {s.CpuUsage,5:F1}%  RAM {s.RamUsage,5:F1}%  Conn {s.ActiveConnections,5}");
			}
		}

		if (replicas.Count > show)
			Console.WriteLine($"  ... ({replicas.Count - show} more replicas)");

		Console.WriteLine();
		Console.WriteLine("Recent events:");
		var events = cluster.Events;
		var last = Math.Min(8, events.Count);
		for (var i = events.Count - last; i < events.Count; i++)
			Console.WriteLine($"  {events[i].TimeStampUtc:HH:mm:ss} {events[i].Message}");
	}

	public static void PrintState(
		ServerDigitalTwin twin,
		ServerMetricsReading reading,
		ServerLoadPredictionOutput scale,
		ServerSecurityOutput sec)
	{
		Console.Clear();

		var s = twin.State;

		Console.WriteLine($"Time: {reading.TimeStamp:HH:mm:ss}");
		Console.WriteLine($"Server: {s.ServerId} ({s.Role}), Status: {s.Status}");
		Console.WriteLine($"CPU: {s.CpuUsage:F1}%   RAM: {s.RamUsage:F1}%   Disk: {s.DiskUsage:F1}%");
		Console.WriteLine($"Net: In {s.NetInMbps:F1} Mbps  Out {s.NetOutMbps:F1} Mbps");
		Console.WriteLine($"Conn: {s.ActiveConnections}   Errors/min: {s.ErrorRatePerMinute:F1}");
		Console.WriteLine($"Temp: {s.TemperatureC:F1} °C");
		Console.WriteLine();

		Console.WriteLine($"[AUTOSCALE] risk={scale.OverloadRisk:F2}, replicaDelta={scale.RecommendedReplicaDelta}");
		Console.WriteLine($"[SECURITY]  anomalyScore={sec.AnomalyScore:F2}, isAnomaly={sec.IsAnomaly}, " +
						  $"DDoS={sec.IsSuspectedDdos}, Bruteforce={sec.IsSuspectedBruteForce}");
		Console.WriteLine();
		Console.WriteLine("Press Ctrl+C to stop...");
	}

	private static string Bar(double value, int width = 18)
	{
		value = ClampHelper.Clamp(value, 0, 100);
		var filled = (int)Math.Round((value / 100.0) * width);
		filled = Math.Clamp(filled, 0, width);
		var empty = width - filled;
		return $"[{new string('█', filled)}{new string('░', empty)}] {value,5:F1}%";
	}
}