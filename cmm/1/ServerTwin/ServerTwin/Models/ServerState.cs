namespace ServerTwin.Models;

public partial class ServerState(string serverId, ServerRole role)
{
	public string ServerId { get; } = serverId;
	public ServerRole Role { get; } = role;
	public ServerStatus Status { get; private set; } = ServerStatus.Unknown;
	public double CpuUsage { get; private set; }
	public double RamUsage { get; private set; }
	public double DiskUsage { get; private set; }
	public double DiskIOPs { get; private set; }
	public double NetInMbps { get; private set; }
	public double NetOutMbps { get; private set; }
	public double TemperatureC { get; private set; }
	public int ActiveConnections { get; private set; }
	public double ErrorRatePerMinute { get; private set; }
	public DateTime LastUpdated { get; private set; }

	public bool IsCpuOverloaded => CpuUsage > 85.0;
	public bool IsRamOverloaded => RamUsage > 90.0;
	public bool IsDiskCritical => DiskUsage > 95.0;
	public bool IsHighErrorRate => ErrorRatePerMinute > 30.0;
}

public partial class ServerState
{
	public void UpdateFrom(ServerMetricsReading reading)
	{
		CpuUsage = reading.CpuUsagePercentage;
		RamUsage = reading.RamUsagePercentage;
		DiskUsage = reading.DiskUsagePercentage;
		DiskIOPs = reading.DiskIOPs;
		NetInMbps = reading.NetInMbps;
		NetOutMbps = reading.NetOutMbps;
		TemperatureC = reading.TemperatureC;
		ActiveConnections = reading.ActiveConnections;
		ErrorRatePerMinute = reading.ErrorRatePerMinute;
		LastUpdated = reading.TimeStamp;

		RecalculateStatus();
	}

	private void  RecalculateStatus()
	{
		if ((DateTime.UtcNow - LastUpdated).TotalMinutes > 5)
			Status = ServerStatus.Unknown;
		else if (IsCpuOverloaded || IsRamOverloaded || IsDiskCritical || IsHighErrorRate)
			Status = ServerStatus.Critical;
		else if (CpuUsage > 70.0 || RamUsage > 75.0 || DiskUsage > 80.0)
			Status = ServerStatus.Degraded;
		else
			Status = ServerStatus.Healthy;
	}
}