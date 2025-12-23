namespace ServerTwin.Helpers;

public static class ClampHelper
{
	public static double Clamp01(double value)
	{
		if (value < 0) return 0;
		if (value > 1) return 1;
		return value;
	}

	public static double Clamp(double value, double min, double max)
	{
		if (value < min) return min;
		if (value > max) return max;
		return value;
	}
}