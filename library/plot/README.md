# Plot

`plot` provides explicit, composable charts over Severian's `graphics` package.
It has no hidden current figure: every series and label belongs to a `Chart`.

```sev
import plot

chart := plot.Chart(800, 500)
_line ?= chart.line_labeled([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], "x squared")
chart.title("A useful curve")
chart.x_label("x")
chart.y_label("y")
_saved ?= chart.save("curve.svg")
```

The initial implementation supports line, scatter, bar, and histogram series.
`line_data` consumes named `Data` columns, and `line_tensor` consumes
`Tensor[f64]` values without requiring callers to manually convert them.

`render()` returns a normal `graphics.Canvas`, so applications may add custom
annotations before saving. Interactive `show()` will arrive with the graphics
window/event backend; headless `save()` is the deterministic baseline.
