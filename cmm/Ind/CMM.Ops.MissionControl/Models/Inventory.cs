using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Models;

public sealed class Inventory : ObservableObject
{
    private int _food;
    private int _parts;
    private int _ammo;

    public int MaxFood { get; } = 100;
    public int MaxParts { get; } = 40;
    public int MaxAmmo { get; } = 80;

    public int Food
    {
        get => _food;
        set => Set(ref _food, Clamp(value, 0, MaxFood));
    }

    public int Parts
    {
        get => _parts;
        set => Set(ref _parts, Clamp(value, 0, MaxParts));
    }

    public int Ammo
    {
        get => _ammo;
        set => Set(ref _ammo, Clamp(value, 0, MaxAmmo));
    }

    public Inventory()
    {
        Food = 80;
        Parts = 20;
        Ammo = 40;
    }

    public void Reset()
    {
        Food = 80;
        Parts = 20;
        Ammo = 40;
    }

    public bool TryConsumeParts(int count)
    {
        if (count <= 0) return true;
        if (Parts < count) return false;
        Parts -= count;
        return true;
    }

    public void ConsumeFood(int count) => Food -= count;
    public void ConsumeAmmo(int count) => Ammo -= count;

    private static int Clamp(int v, int min, int max)
    {
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }
}
