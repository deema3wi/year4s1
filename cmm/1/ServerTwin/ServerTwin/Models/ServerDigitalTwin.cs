namespace ServerTwin.Models;

public partial class ServerDigitalTwin(string serverId, ServerRole role, int maxHistorySize = 2048)
{
	public ServerState State { get; } = new(serverId, role);
	private readonly Queue<ServerMetricsReading> _history = new();
	private readonly int _maxHistorySize = maxHistorySize;
}

public partial class ServerDigitalTwin
{
	public void ApplyMetrics(ServerMetricsReading reading)
	{
		State.UpdateFrom(reading);

		if (_history.Count >= _maxHistorySize) return;
		_history.Enqueue(reading);
	}

	public IReadOnlyCollection<ServerMetricsReading> GetHistory() => _history.ToArray().AsReadOnly();
}