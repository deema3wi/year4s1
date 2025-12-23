namespace ServerTwin.Services;

public partial class ServerAutoScalingAdvisorService(ServerDigitalTwin twin,
	IAIModel<ServerLoadPredictionInput, ServerLoadPredictionOutput> model)
{
	private readonly ServerDigitalTwin _twin = twin;
	private readonly IAIModel<ServerLoadPredictionInput, ServerLoadPredictionOutput> _model = model;
}

public partial class ServerAutoScalingAdvisorService
{
	public ServerLoadPredictionOutput EvaluateScaling()
	{
		var s = _twin.State;
		ServerLoadPredictionInput input = new(
			s.CpuUsage,
			s.RamUsage,
			s.NetInMbps,
			s.NetOutMbps,
			s.ActiveConnections
		);

		ServerLoadPredictionOutput output = _model.Predict(input);
		if (output.OverloadRisk > 0.7f && output.RecommendedReplicaDelta > 0)
		{
			Console.WriteLine($"[AUTOSCALE] Overload Risk={output.OverloadRisk:F2}{Environment.NewLine}" +
				$"add {output.RecommendedReplicaDelta} replicas");
		}

		return output;
	}
}