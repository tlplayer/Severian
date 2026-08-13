# Graphics and plotting

These examples exercise Severian's deterministic, headless visualization stack.
They write SVG files under `/tmp`, so they run without a display server and use
the same output path in local development and CI.

- `01-graphics.sev` draws directly into an explicit `graphics.Canvas`.
- `02-plot.sev` turns named `Data` columns into a labeled `plot.Chart` and then
  renders it through `graphics`.

Run them with:

```sh
sev run docs/examples/29-visualization/01-graphics.sev
sev run docs/examples/29-visualization/02-plot.sev
```

The generated files are `/tmp/severian-graphics-example.svg` and
`/tmp/severian-plot-example.svg`.
