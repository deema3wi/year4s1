namespace ServerTwin.Models;

public sealed record ClusterEvent(DateTime TimeStampUtc, string Message);
