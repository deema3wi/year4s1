using System;
using System.Collections.Generic;
using CMM.Ops.MissionControl.Infrastructure;

namespace CMM.Ops.MissionControl.Models;

public sealed class MapGrid : ObservableObject
{
    private readonly MapCell[,] _cells;

    public int Width { get; }
    public int Height { get; }

    public event EventHandler? Changed;

    public MapGrid(int width, int height)
    {
        if (width < 8 || height < 8) throw new ArgumentOutOfRangeException("Map too small.");
        Width = width;
        Height = height;

        _cells = new MapCell[width, height];
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            var cell = new MapCell(CellType.Empty, 0);
            cell.PropertyChanged += (_, __) => Changed?.Invoke(this, EventArgs.Empty);
            _cells[x, y] = cell;
        }
    }

    public MapCell this[int x, int y] => _cells[x, y];

    public bool InBounds(int x, int y) => x >= 0 && y >= 0 && x < Width && y < Height;

    public void Clear()
    {
        for (var y = 0; y < Height; y++)
        for (var x = 0; x < Width; x++)
        {
            _cells[x, y].Type = CellType.Empty;
            _cells[x, y].Risk = 0;
        }
        Changed?.Invoke(this, EventArgs.Empty);
    }

    public void GenerateRandom(int seed, double obstacleDensity = 0.18, double riskDensity = 0.35)
    {
        var rng = new Random(seed);

        Clear();

        // Border obstacles (soft border) to look like a "playfield"
        for (var x = 0; x < Width; x++)
        {
            if (rng.NextDouble() < 0.65) _cells[x, 0].Type = CellType.Obstacle;
            if (rng.NextDouble() < 0.65) _cells[x, Height - 1].Type = CellType.Obstacle;
        }
        for (var y = 0; y < Height; y++)
        {
            if (rng.NextDouble() < 0.65) _cells[0, y].Type = CellType.Obstacle;
            if (rng.NextDouble() < 0.65) _cells[Width - 1, y].Type = CellType.Obstacle;
        }

        for (var y = 1; y < Height - 1; y++)
        for (var x = 1; x < Width - 1; x++)
        {
            if (rng.NextDouble() < obstacleDensity)
            {
                _cells[x, y].Type = CellType.Obstacle;
                continue;
            }

            if (rng.NextDouble() < riskDensity)
            {
                // clustered-ish risk
                var baseRisk = rng.NextDouble() * rng.NextDouble(); // bias toward small values
                _cells[x, y].Risk = Math.Clamp(baseRisk * 1.2, 0, 1);
            }
        }

        // Make a few "danger hotspots"
        for (int i = 0; i < 3; i++)
        {
            var cx = rng.Next(2, Width - 2);
            var cy = rng.Next(2, Height - 2);
            for (var dy = -3; dy <= 3; dy++)
            for (var dx = -3; dx <= 3; dx++)
            {
                var x = cx + dx;
                var y = cy + dy;
                if (!InBounds(x, y)) continue;
                if (_cells[x, y].Type == CellType.Obstacle) continue;

                var d = Math.Sqrt(dx * dx + dy * dy);
                var add = Math.Clamp(1.0 - d / 3.5, 0, 1);
                _cells[x, y].Risk = Math.Clamp(_cells[x, y].Risk + add * 0.7, 0, 1);
            }
        }

        Changed?.Invoke(this, EventArgs.Empty);
    }

    public IEnumerable<GridPoint> Neighbors4(GridPoint p)
    {
        var (x, y) = p;
        if (InBounds(x + 1, y)) yield return new GridPoint(x + 1, y);
        if (InBounds(x - 1, y)) yield return new GridPoint(x - 1, y);
        if (InBounds(x, y + 1)) yield return new GridPoint(x, y + 1);
        if (InBounds(x, y - 1)) yield return new GridPoint(x, y - 1);
    }
}
