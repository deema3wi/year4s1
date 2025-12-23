namespace ServerTwin.Services;

public sealed class ClusterCapacityPlanningService
{
	private readonly ServerClusterTwin _cluster;
	private readonly IAIModel<CapacityPlanningInput, CapacityPlanningOutput> _planningModel;

	public ClusterCapacityPlanningService(
		ServerClusterTwin cluster,
		IAIModel<CapacityPlanningInput, CapacityPlanningOutput> planningModel)
	{
		_cluster = cluster;
		_planningModel = planningModel;
	}

	public CapacityPlanningOutput Plan()
	{
		var history = _cluster.GetClusterHistory();
		CapacityPlanningInput input = new(
			ServerId: "cluster",
			Role: _cluster.Role,
			History: history
		);

		return _planningModel.Predict(input);
	}
}
