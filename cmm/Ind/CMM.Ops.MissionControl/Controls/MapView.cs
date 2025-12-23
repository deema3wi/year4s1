using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using CMM.Ops.MissionControl.Models;

namespace CMM.Ops.MissionControl;

public sealed class MapView : FrameworkElement
{
    public static readonly DependencyProperty MapProperty =
        DependencyProperty.Register(nameof(Map), typeof(MapGrid), typeof(MapView),
            new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender, OnMapChanged));

    public static readonly DependencyProperty PathProperty =
        DependencyProperty.Register(nameof(Path), typeof(IReadOnlyList<GridPoint>), typeof(MapView),
            new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty DronePositionProperty =
        DependencyProperty.Register(nameof(DronePosition), typeof(GridPoint), typeof(MapView),
            new FrameworkPropertyMetadata(default(GridPoint), FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty TargetsProperty =
        DependencyProperty.Register(nameof(Targets), typeof(ObservableCollection<GridPoint>), typeof(MapView),
            new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty IsEditModeProperty =
        DependencyProperty.Register(nameof(IsEditMode), typeof(bool), typeof(MapView),
            new FrameworkPropertyMetadata(false));

    public static readonly DependencyProperty CellSizeProperty =
        DependencyProperty.Register(nameof(CellSize), typeof(double), typeof(MapView),
            new FrameworkPropertyMetadata(18.0, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty OnLogProperty =
        DependencyProperty.Register(nameof(OnLog), typeof(ICommand), typeof(MapView),
            new FrameworkPropertyMetadata(null));

    public MapGrid? Map
    {
        get => (MapGrid?)GetValue(MapProperty);
        set => SetValue(MapProperty, value);
    }

    public IReadOnlyList<GridPoint>? Path
    {
        get => (IReadOnlyList<GridPoint>?)GetValue(PathProperty);
        set => SetValue(PathProperty, value);
    }

    public GridPoint DronePosition
    {
        get => (GridPoint)GetValue(DronePositionProperty);
        set => SetValue(DronePositionProperty, value);
    }

    public ObservableCollection<GridPoint>? Targets
    {
        get => (ObservableCollection<GridPoint>?)GetValue(TargetsProperty);
        set => SetValue(TargetsProperty, value);
    }

    public bool IsEditMode
    {
        get => (bool)GetValue(IsEditModeProperty);
        set => SetValue(IsEditModeProperty, value);
    }

    public double CellSize
    {
        get => (double)GetValue(CellSizeProperty);
        set => SetValue(CellSizeProperty, value);
    }

    public ICommand? OnLog
    {
        get => (ICommand?)GetValue(OnLogProperty);
        set => SetValue(OnLogProperty, value);
    }

    private static void OnMapChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is not MapView v) return;

        if (e.OldValue is MapGrid oldMap)
            oldMap.Changed -= v.OnMapInternalChanged;

        if (e.NewValue is MapGrid newMap)
            newMap.Changed += v.OnMapInternalChanged;

        v.InvalidateVisual();
        v.InvalidateMeasure();
    }

    private void OnMapInternalChanged(object? sender, EventArgs e) => Dispatcher.Invoke(InvalidateVisual);

    protected override Size MeasureOverride(Size availableSize)
    {
        if (Map is null) return new Size(200, 200);

        // If CellSize is set (>0), keep a stable pixel size.
        if (CellSize > 0.0)
            return new Size(Map.Width * CellSize + 1, Map.Height * CellSize + 1);

        // Auto-fit mode (CellSize <= 0): request all available space.
        // If parent gives infinity, fallback to a reasonable default.
        if (double.IsInfinity(availableSize.Width) || double.IsInfinity(availableSize.Height))
            return new Size(Map.Width * 18 + 1, Map.Height * 18 + 1);

        return availableSize;
    }

    protected override void OnRender(DrawingContext dc)
    {
        base.OnRender(dc);

        dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(11, 13, 18)), null, new Rect(0, 0, ActualWidth, ActualHeight));

        if (Map is null) return;

        var cs = GetEffectiveCellSize();
        var origin = GetOrigin(cs);
        var gridPen = new Pen(new SolidColorBrush(Color.FromRgb(34, 40, 56)), 1);

        for (var y = 0; y < Map.Height; y++)
        for (var x = 0; x < Map.Width; x++)
        {
            var rect = new Rect(origin.X + x * cs, origin.Y + y * cs, cs, cs);
            var cell = Map[x, y];

            if (cell.Type == CellType.Obstacle)
            {
                dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(35, 39, 52)), null, rect);
            }
            else
            {
                // risk heatmap overlay
                if (cell.Risk > 0.001)
                {
                    var alpha = (byte)Math.Clamp(cell.Risk * 200, 0, 200);
                    var riskColor = Color.FromArgb(alpha, 210, 64, 64);
                    dc.DrawRectangle(new SolidColorBrush(riskColor), null, rect);
                }
            }

            dc.DrawRectangle(null, gridPen, rect);
        }

        // Path overlay
        if (Path is { Count: > 1 })
        {
            var pathPen = new Pen(new SolidColorBrush(Color.FromRgb(107, 181, 255)), 2);
            for (int i = 1; i < Path.Count; i++)
            {
                var a = Path[i - 1];
                var b = Path[i];
                dc.DrawLine(pathPen, Center(a), Center(b));
            }
        }

        // Targets
        if (Targets is not null)
        {
            foreach (var t in Targets)
            {
                var p = Center(t);
                dc.DrawEllipse(new SolidColorBrush(Color.FromRgb(255, 214, 99)), new Pen(new SolidColorBrush(Color.FromRgb(34, 40, 56)), 1), p, cs * 0.25, cs * 0.25);
            }
        }

        // Drone
        {
            var p = Center(DronePosition);
            dc.DrawEllipse(new SolidColorBrush(Color.FromRgb(92, 255, 179)), new Pen(new SolidColorBrush(Color.FromRgb(34, 40, 56)), 1), p, cs * 0.28, cs * 0.28);
        }

        Point Center(GridPoint gp) => new(origin.X + (gp.X + 0.5) * cs, origin.Y + (gp.Y + 0.5) * cs);
    }

    protected override void OnMouseDown(MouseButtonEventArgs e)
    {
        base.OnMouseDown(e);

        if (Map is null) return;

        var cs = GetEffectiveCellSize();
        var origin = GetOrigin(cs);
        var p = e.GetPosition(this);

        // click outside the grid -> ignore
        if (p.X < origin.X || p.Y < origin.Y) return;

        var x = (int)((p.X - origin.X) / cs);
        var y = (int)((p.Y - origin.Y) / cs);
        if (!Map.InBounds(x, y)) return;

        var gp = new GridPoint(x, y);

        if (!IsEditMode)
        {
            var cell = Map[x, y];
            Log($"Cell {gp}: type={cell.Type}, risk={cell.Risk:0.00}");
            return;
        }

        if (e.ChangedButton == MouseButton.Left)
        {
            var cell = Map[x, y];
            cell.Type = cell.Type == CellType.Obstacle ? CellType.Empty : CellType.Obstacle;
            if (cell.Type == CellType.Obstacle) cell.Risk = 0;
            Log($"Edit: toggle obstacle at {gp} -> {cell.Type}");
            return;
        }

        if (e.ChangedButton == MouseButton.Right)
        {
            var cell = Map[x, y];
            if (cell.Type == CellType.Obstacle) return;

            // cycle risk: 0 -> 0.35 -> 0.7 -> 1 -> 0
            var r = cell.Risk;
            var next = r switch
            {
                < 0.05 => 0.35,
                < 0.5 => 0.7,
                < 0.85 => 1.0,
                _ => 0.0
            };
            cell.Risk = next;
            Log($"Edit: risk at {gp} -> {cell.Risk:0.00}");
            return;
        }
    }

    private double GetEffectiveCellSize()
    {
        if (Map is null) return CellSize > 0 ? CellSize : 18.0;
        if (CellSize > 0.0) return CellSize;

        // Fit grid to the available space.
        var w = Math.Max(1.0, ActualWidth);
        var h = Math.Max(1.0, ActualHeight);
        var csW = (w - 2) / Math.Max(1, Map.Width);
        var csH = (h - 2) / Math.Max(1, Map.Height);
        var cs = Math.Floor(Math.Min(csW, csH));
        return Math.Max(8.0, cs);
    }

    private Point GetOrigin(double cs)
    {
        if (Map is null) return new Point(0, 0);
        var gridW = Map.Width * cs;
        var gridH = Map.Height * cs;
        var ox = (ActualWidth - gridW) / 2.0;
        var oy = (ActualHeight - gridH) / 2.0;
        return new Point(Math.Max(0, ox), Math.Max(0, oy));
    }

    private void Log(string message)
    {
        if (OnLog is null) return;
        if (OnLog.CanExecute(message))
            OnLog.Execute(message);
    }
}
