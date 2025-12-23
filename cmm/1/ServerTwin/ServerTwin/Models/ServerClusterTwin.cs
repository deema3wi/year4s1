namespace ServerTwin.Models;

public sealed class ServerClusterTwin
{
	private readonly List<ServerDigitalTwin> _replicas = new();
	private readonly Queue<ServerMetricsReading> _clusterHistory = new();
	private readonly int _maxHistorySize;
	private int _nextReplicaIndex = 1;

	public ServerRole Role { get; }
	public IReadOnlyList<ServerDigitalTwin> Replicas => _replicas;
	public List<ClusterEvent> Events { get; } = new(64);
	public ClusterMetrics LastClusterMetrics { get; private set; } = new(
		TimeStampUtc: DateTime.UtcNow,
		Replicas: 0,
		AvgCpu: 0,
		AvgRam: 0,
		AvgDisk: 0,
		AvgDiskIOPs: 0,
		AvgNetInMbps: 0,
		AvgNetOutMbps: 0,
		AvgConnections: 0,
		AvgErrorRatePerMinute: 0,
		AvgTemperatureC: 0
	);

	public ServerClusterTwin(int initialReplicas, ServerRole role, int maxHistorySize = 2048)
	{
		if (initialReplicas <= 0) throw new ArgumentOutOfRangeException(nameof(initialReplicas));
		Role = role;
		_maxHistorySize = maxHistorySize;
		ScaleOut(initialReplicas);
	}

	public int ReplicaCount => _replicas.Count;

	public IReadOnlyCollection<ServerMetricsReading> GetClusterHistory() => _clusterHistory.ToArray().AsReadOnly();

	public void ApplyWorkload(WorkloadReading workload, Random rnd)
	{
		if (_replicas.Count == 0) return;

		var r = _replicas.Count;

		for (var i = 0; i < _replicas.Count; i++)
		{
			var per = ToPerReplicaReading(workload, r, rnd);
			_replicas[i].ApplyMetrics(per);
		}

		var agg = Aggregate(workload.TimeStamp, _replicas);
		LastClusterMetrics = agg;

		if (_clusterHistory.Count < _maxHistorySize)
		{
			_clusterHistory.Enqueue(new ServerMetricsReading(
				TimeStamp: workload.TimeStamp,
				CpuUsagePercentage: agg.AvgCpu,
				RamUsagePercentage: agg.AvgRam,
				DiskUsagePercentage: agg.AvgDisk,
				DiskIOPs: agg.AvgDiskIOPs,
				NetInMbps: agg.AvgNetInMbps,
				NetOutMbps: agg.AvgNetOutMbps,
				TemperatureC: agg.AvgTemperatureC,
				ActiveConnections: agg.AvgConnections,
				ErrorRatePerMinute: agg.AvgErrorRatePerMinute
			));
		}
	}

	public void ScaleOut(int count)
	{
		if (count <= 0) return;
		for (var i = 0; i < count; i++)
		{
			var id = $"srv-{_nextReplicaIndex++}";
			_replicas.Add(new ServerDigitalTwin(id, Role));
		}
	}

	public void ScaleIn(int count)
	{
		if (count <= 0) return;
		count = Math.Min(count, _replicas.Count);
		for (var i = 0; i < count; i++)
			_replicas.RemoveAt(_replicas.Count - 1);
	}

	public ServerLoadPredictionInput GetLoadInputPerReplica()
	{
		var m = LastClusterMetrics;
		return new ServerLoadPredictionInput(
			CurrentCpuUsage: m.AvgCpu,
			CurrentMemoryUsage: m.AvgRam,
			NetInMbps: m.AvgNetInMbps,
			NetOutMbps: m.AvgNetOutMbps,
			ActiveConnections: m.AvgConnections
		);
	}

	public ServerSecurityInput GetSecurityInputPerReplica()
	{
		var m = LastClusterMetrics;
		return new ServerSecurityInput(
			CpuUsage: m.AvgCpu,
			NetInMbps: m.AvgNetInMbps,
			NetOutMbps: m.AvgNetOutMbps,
			ActiveConnections: m.AvgConnections,
			ErrorRatePerMinute: m.AvgErrorRatePerMinute
		);
	}

	private static ServerMetricsReading ToPerReplicaReading(WorkloadReading w, int replicas, Random rnd)
	{
		var cpuBase = w.CpuDemand / replicas;
		var cpu = ClampHelper.Clamp(cpuBase + (rnd.NextDouble() * 2 - 1) * 6, 0, 100);

		var ramBase = w.RamDemand / replicas;
		var ram = ClampHelper.Clamp(ramBase + (rnd.NextDouble() * 2 - 1) * 5, 0, 100);

		var disk = ClampHelper.Clamp(w.DiskBaseUsagePercentage + (rnd.NextDouble() * 2 - 1) * 0.8, 0, 100);
		var diskIops = Math.Max(0, (w.DiskIopsDemand / replicas) + rnd.Next(0, 90));

		var netIn = Math.Max(0, (w.NetInDemandMbps / replicas) + rnd.NextDouble() * 20);
		var netOut = Math.Max(0, (w.NetOutDemandMbps / replicas) + rnd.NextDouble() * 12);

		var conns = (int)Math.Max(0, (w.ConnectionDemand / (double)replicas) + rnd.Next(-35, 36));

		var errBase = w.ErrorRateDemandPerMinute / replicas;
		var err = cpu > 92 ? errBase + rnd.Next(5, 30) : errBase + rnd.Next(0, 6);
		err = Math.Max(0, err);

		var temp = 30 + cpu * 0.35 + (rnd.NextDouble() * 2 - 1) * 0.6;

		return new ServerMetricsReading(
			TimeStamp: w.TimeStamp,
			CpuUsagePercentage: cpu,
			RamUsagePercentage: ram,
			DiskUsagePercentage: disk,
			DiskIOPs: diskIops,
			NetInMbps: netIn,
			NetOutMbps: netOut,
			TemperatureC: temp,
			ActiveConnections: conns,
			ErrorRatePerMinute: err
		);
	}

	private static ClusterMetrics Aggregate(DateTime ts, List<ServerDigitalTwin> replicas)
	{
		var r = replicas.Count;
		if (r == 0)
			return new ClusterMetrics(ts, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

		double sumCpu = 0, sumRam = 0, sumDisk = 0, sumIops = 0, sumIn = 0, sumOut = 0, sumErr = 0, sumTemp = 0;
		int sumConn = 0;

		foreach (var t in replicas)
		{
			var s = t.State;
			sumCpu += s.CpuUsage;
			sumRam += s.RamUsage;
			sumDisk += s.DiskUsage;
			sumIops += s.DiskIOPs;
			sumIn += s.NetInMbps;
			sumOut += s.NetOutMbps;
			sumConn += s.ActiveConnections;
			sumErr += s.ErrorRatePerMinute;
			sumTemp += s.TemperatureC;
		}

		return new ClusterMetrics(
			TimeStampUtc: ts,
			Replicas: r,
			AvgCpu: sumCpu / r,
			AvgRam: sumRam / r,
			AvgDisk: sumDisk / r,
			AvgDiskIOPs: sumIops / r,
			AvgNetInMbps: sumIn / r,
			AvgNetOutMbps: sumOut / r,
			AvgConnections: (int)Math.Round(sumConn / (double)r),
			AvgErrorRatePerMinute: sumErr / r,
			AvgTemperatureC: sumTemp / r
		);
	}
}
