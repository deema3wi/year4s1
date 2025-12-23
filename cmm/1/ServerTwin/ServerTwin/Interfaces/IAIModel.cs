namespace ServerTwin.Interfaces;

public interface IAIModel<TInput, TOutput>
{
	TOutput Predict(TInput input);
}