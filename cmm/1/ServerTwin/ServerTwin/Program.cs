using ServerTwin.Services;

var cluster = new ServerClusterTwin(initialReplicas: 1, role: ServerRole.Web);

var loadModel = new ServerLoadModel();
var securityModel = new ServerSecurityModel();
var planningModel = new CapacityPlanningModel();

var autoscaler = new ServerAutoScalerService(
	cluster,
	loadModel,
	new AutoScalingPolicy(
		MinReplicas: 1,
		MaxReplicas: 10,
		Cooldown: TimeSpan.FromSeconds(10),
		LowRiskHoldTime: TimeSpan.FromSeconds(12))
);

var security = new ClusterSecurityMonitoringService(cluster, securityModel);
var capacity = new ClusterCapacityPlanningService(cluster, planningModel);

var rnd = new Random();
var lastPlanningAt = DateTime.UtcNow;

var workloadPreset = SimulationPreset.Realistic;
var visualPreset = VisualizationPreset.Dynamic;

for (int step = 0; step < 600; step++)
{
	var now = DateTime.UtcNow;
	HandleKeys(ref workloadPreset, ref visualPreset);

	var workload = WorkloadGenerator.Generate(rnd, now, workloadPreset);
	cluster.ApplyWorkload(workload, rnd);

	var scale = autoscaler.EvaluateAndApply(now);
	var sec = security.Analyze();

	if ((now - lastPlanningAt).TotalSeconds >= 30)
	{
		var plan = capacity.Plan();
		cluster.Events.Add(new ClusterEvent(now, $"[CAPACITY] {plan.Recommendation}"));
		lastPlanningAt = now;
	}

	PrintHelper.PrintCluster(cluster, workload, scale, sec, workloadPreset, visualPreset);
	Thread.Sleep(1000);
}

Console.WriteLine("Simulation finished. Press any key to exit.");
Console.ReadKey();

static void HandleKeys(ref SimulationPreset workloadPreset, ref VisualizationPreset visualPreset)
{
	if (!Console.KeyAvailable) return;
	var key = Console.ReadKey(intercept: true);

	switch (key.Key)
	{
		case ConsoleKey.R:
			workloadPreset = SimulationPreset.Realistic;
			break;
		case ConsoleKey.D:
			workloadPreset = SimulationPreset.Dynamic;
			break;
		case ConsoleKey.V:
			visualPreset = visualPreset == VisualizationPreset.Realistic
				? VisualizationPreset.Dynamic
				: VisualizationPreset.Realistic;
			break;
		case ConsoleKey.Q:
			Environment.Exit(0);
			break;
	}
}
