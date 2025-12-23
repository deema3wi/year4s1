namespace ServerTwin.Services;

public partial class CapacityPlanningService(ServerDigitalTwin twin,
	IAIModel<CapacityPlanningInput, CapacityPlanningOutput> planningModel)
{
	private readonly ServerDigitalTwin _twin = twin;
	private readonly IAIModel<CapacityPlanningInput, CapacityPlanningOutput> _planningModel = planningModel;
}

public partial class CapacityPlanningService
{
	public CapacityPlanningOutput Plan()
	{
		var history = _twin.GetHistory();
		var s = _twin.State;
		CapacityPlanningInput input = new(
			s.ServerId,
			s.Role,
			history
		);

		var output = _planningModel.Predict(input);
		Console.WriteLine($"[CAPACITY] Recommendation: {output.Recommendation}");
		return output;
	}
}