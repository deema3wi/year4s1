namespace ServerTwin.AIModels;

public class ServerLoadModel : IAIModel<ServerLoadPredictionInput, ServerLoadPredictionOutput>
{
	public ServerLoadPredictionOutput Predict(ServerLoadPredictionInput input)
	{
		var avgUsage = (input.CurrentCpuUsage + input.CurrentMemoryUsage) / 2.0d;
		var baseRisk = ClampHelper.Clamp01(avgUsage / 100.0d);

		var connectionFactor = ClampHelper.Clamp01(input.ActiveConnections / 1000.0d);
		var netFactor = ClampHelper.Clamp01((input.NetInMbps + input.NetOutMbps) / 200.0d);

		var risk = baseRisk * 0.6 + connectionFactor * 0.2 + netFactor * 0.2;
		risk = ClampHelper.Clamp01(risk);
		int replicaDelta = risk switch
		{
			>= 0.8d => 2,
			>= 0.6d and < 0.8d => 1,
			>= 0.2d and < 0.6d => 0,
			_ => -1
		};

		return new((float)risk, replicaDelta);
	}
}