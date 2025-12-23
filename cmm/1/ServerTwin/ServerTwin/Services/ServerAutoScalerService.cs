namespace ServerTwin.Services;

public sealed class ServerAutoScalerService
{
	private readonly ServerClusterTwin _cluster;
	private readonly IAIModel<ServerLoadPredictionInput, ServerLoadPredictionOutput> _model;
	private readonly AutoScalingPolicy _policy;

	private DateTime _lastScaleAt = DateTime.MinValue;
	private DateTime? _lowRiskSince = null;

	public ServerAutoScalerService(
		ServerClusterTwin cluster,
		IAIModel<ServerLoadPredictionInput, ServerLoadPredictionOutput> model,
		AutoScalingPolicy policy)
	{
		_cluster = cluster;
		_model = model;
		_policy = policy;
	}

	public AutoScalingResult EvaluateAndApply(DateTime nowUtc)
	{
		var input = _cluster.GetLoadInputPerReplica();
		var output = _model.Predict(input);

		var risk = output.OverloadRisk;
		var replicasBefore = _cluster.ReplicaCount;
		var appliedDelta = 0;
		string reason = "";

		var cooldownLeft = (_lastScaleAt == DateTime.MinValue)
			? TimeSpan.Zero
			: (_policy.Cooldown - (nowUtc - _lastScaleAt));

		if (cooldownLeft > TimeSpan.Zero)
		{
			reason = $"cooldown ({cooldownLeft.TotalSeconds:F0}s)";
			return Build(nowUtc, output, appliedDelta, replicasBefore, reason);
		}

		if (risk <= _policy.LowRiskThreshold)
			_lowRiskSince ??= nowUtc;
		else
			_lowRiskSince = null;

		if (output.RecommendedReplicaDelta > 0 && risk >= _policy.HighRiskThreshold)
		{
			var canAdd = _policy.MaxReplicas - replicasBefore;
			var wantAdd = Math.Min(output.RecommendedReplicaDelta, _policy.MaxScaleOutPerAction);
			var add = Math.Min(canAdd, wantAdd);

			if (add > 0)
			{
				_cluster.ScaleOut(add);
				appliedDelta = add;
				_lastScaleAt = nowUtc;
				reason = $"scale out +{add} (risk={risk:F2})";
				_cluster.Events.Add(new ClusterEvent(nowUtc, $"[AUTOSCALE] {reason}"));
			}
			else
			{
				reason = "max replicas reached";
			}
		}
		
		else if (output.RecommendedReplicaDelta < 0 && replicasBefore > _policy.MinReplicas)
		{
			var lowRiskFor = _lowRiskSince is null ? TimeSpan.Zero : (nowUtc - _lowRiskSince.Value);
			if (_lowRiskSince is not null && lowRiskFor >= _policy.LowRiskHoldTime)
			{
				var canRemove = replicasBefore - _policy.MinReplicas;
				var remove = Math.Min(canRemove, _policy.MaxScaleInPerAction);
				if (remove > 0)
				{
					_cluster.ScaleIn(remove);
					appliedDelta = -remove;
					_lastScaleAt = nowUtc;
					reason = $"scale in {appliedDelta} (low risk {lowRiskFor.TotalSeconds:F0}s, risk={risk:F2})";
					_cluster.Events.Add(new ClusterEvent(nowUtc, $"[AUTOSCALE] {reason}"));
					_lowRiskSince = null;
				}
				else
				{
					reason = "min replicas reached";
				}
			}
			else
			{
				reason = $"low risk hold {lowRiskFor.TotalSeconds:F0}s / {_policy.LowRiskHoldTime.TotalSeconds:F0}s";
			}
		}
		else
		{
			reason = "no scaling";
		}

		return Build(nowUtc, output, appliedDelta, replicasBefore, reason);
	}

	private AutoScalingResult Build(DateTime nowUtc, ServerLoadPredictionOutput output, int appliedDelta, int replicasBefore, string reason)
	{
		return new AutoScalingResult(
			TimeStampUtc: nowUtc,
			Risk: output.OverloadRisk,
			ModelDelta: output.RecommendedReplicaDelta,
			AppliedDelta: appliedDelta,
			ReplicasBefore: replicasBefore,
			ReplicasAfter: _cluster.ReplicaCount,
			Reason: reason
		);
	}
}
