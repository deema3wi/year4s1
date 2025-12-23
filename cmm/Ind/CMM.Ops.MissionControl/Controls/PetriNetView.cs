using System;
using System.Windows;
using System.Windows.Media;
using CMM.Ops.MissionControl.Petri;

namespace CMM.Ops.MissionControl;

public sealed class PetriNetView : FrameworkElement
{
    public static readonly DependencyProperty NetProperty =
        DependencyProperty.Register(nameof(Net), typeof(PetriNet), typeof(PetriNetView),
            new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender, OnNetChanged));

    public static readonly DependencyProperty TokenScaleProperty =
        DependencyProperty.Register(nameof(TokenScale), typeof(double), typeof(PetriNetView),
            new FrameworkPropertyMetadata(1.0, FrameworkPropertyMetadataOptions.AffectsRender));

    public PetriNet? Net
    {
        get => (PetriNet?)GetValue(NetProperty);
        set => SetValue(NetProperty, value);
    }

    public double TokenScale
    {
        get => (double)GetValue(TokenScaleProperty);
        set => SetValue(TokenScaleProperty, value);
    }

    private static void OnNetChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is not PetriNetView v) return;
        if (e.OldValue is PetriNet oldNet)
            oldNet.PropertyChanged -= v.OnNetChangedInternal;
        if (e.NewValue is PetriNet newNet)
            newNet.PropertyChanged += v.OnNetChangedInternal;
        v.InvalidateVisual();
    }

    private void OnNetChangedInternal(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        => Dispatcher.Invoke(InvalidateVisual);

    protected override void OnRender(DrawingContext dc)
    {
        base.OnRender(dc);

        dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(11, 13, 18)), null, new Rect(0, 0, ActualWidth, ActualHeight));

        if (Net is null) return;

        var placeFill = new SolidColorBrush(Color.FromRgb(15, 17, 23));
        var placeStroke = new Pen(new SolidColorBrush(Color.FromRgb(107, 181, 255)), 2);

        var transFill = new SolidColorBrush(Color.FromRgb(35, 39, 52));
        var transStroke = new Pen(new SolidColorBrush(Color.FromRgb(255, 214, 99)), 2);

        var textBrush = new SolidColorBrush(Color.FromRgb(232, 234, 242));
        var faint = new Pen(new SolidColorBrush(Color.FromRgb(50, 58, 80)), 1);

        // Simple layout in "net coordinates" -> pixels
        // We'll treat X/Y in [0..1] normalized.
        Point Px(double nx, double ny) => new(nx * ActualWidth, ny * ActualHeight);

        // Arcs: draw between place and transition centers
        foreach (var t in Net.Transitions)
        {
            var tp = Px(t.X, t.Y);

            foreach (var a in t.Inputs)
            {
                var pp = Px(a.Place.X, a.Place.Y);
                DrawArrow(dc, pp, tp, faint);
            }

            foreach (var a in t.Outputs)
            {
                var pp = Px(a.Place.X, a.Place.Y);
                DrawArrow(dc, tp, pp, faint);
            }
        }

        // Places
        foreach (var p in Net.Places)
        {
            var pos = Px(p.X, p.Y);
            var r = 34.0;
            dc.DrawEllipse(placeFill, placeStroke, pos, r, r);

            // tokens as small circles
            var tokenCount = Math.Min(6, p.Tokens);
            for (int i = 0; i < tokenCount; i++)
            {
                var angle = i * (Math.PI * 2 / Math.Max(1, tokenCount));
                var tr = 8.0 * TokenScale;
                var tx = pos.X + Math.Cos(angle) * 14.0;
                var ty = pos.Y + Math.Sin(angle) * 14.0;
                dc.DrawEllipse(new SolidColorBrush(Color.FromRgb(92, 255, 179)), null, new Point(tx, ty), tr, tr);
            }

            DrawLabel(dc, p.Name, pos, textBrush, dy: 46);
        }

        // Transitions
        foreach (var t in Net.Transitions)
        {
            var pos = Px(t.X, t.Y);
            var w = 22.0;
            var h = 46.0;
            var rect = new Rect(pos.X - w / 2, pos.Y - h / 2, w, h);

            var enabled = t.IsEnabled();
            var stroke = enabled ? new Pen(new SolidColorBrush(Color.FromRgb(92, 255, 179)), 2.5) : transStroke;

            dc.DrawRoundedRectangle(transFill, stroke, rect, 6, 6);
            DrawLabel(dc, t.Name, pos, textBrush, dy: 46);
        }
    }

    private static void DrawLabel(DrawingContext dc, string text, Point center, Brush brush, double dy)
    {
        var ft = new FormattedText(
            text,
            System.Globalization.CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight,
            new Typeface("Segoe UI"),
            13,
            brush,
            1.0);

        dc.DrawText(ft, new Point(center.X - ft.Width / 2, center.Y + dy));
    }

    private static void DrawArrow(DrawingContext dc, Point from, Point to, Pen pen)
    {
        dc.DrawLine(pen, from, to);

        var dir = from - to;
        dir.Normalize();
        var perp = new Vector(-dir.Y, dir.X);

        var tip = to;
        var a = tip + dir * 10 + perp * 5;
        var b = tip + dir * 10 - perp * 5;

        dc.DrawLine(pen, tip, a);
        dc.DrawLine(pen, tip, b);
    }
}
