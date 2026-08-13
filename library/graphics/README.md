# Graphics

`graphics` is Severian's backend-neutral rendering layer. The initial backend is
a deterministic headless SVG canvas. It provides explicit render targets rather
than package-global drawing state, so multiple canvases can be used safely by
tests, servers, and future concurrent programs.

```sev
import graphics

canvas ?= graphics.canvas(640, 480, graphics.white())
canvas.rectangle(40, 40, 180, 80, graphics.blue())
canvas.circle(320, 240, 60, graphics.red())
canvas.line(0, 0, 639, 479, graphics.black())
_saved ?= canvas.save("drawing.svg")
```

The SVG backend is the portable reference implementation and golden-test
target. Pixel images, window/event handling, and GPU resources will use the same
`Canvas`-or-`Frame` ownership model when their typed platform contracts are
added. Backend-specific handles will not appear in ordinary user code.

Text currently uses SVG's generic sans-serif family. A deterministic bundled
font and raster image target are the next headless-rendering milestones.
