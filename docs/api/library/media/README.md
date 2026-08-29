# Media libraries

Stable IDs: `library.media.audio`, `library.media.graphics`, and
`library.media.plot`.

Audio owns sample/container transformations, graphics owns image/pixel
operations, and plot owns visualization composition. Tensor may implement the
numeric work, but a tensor does not by itself define sample rate, channel
layout, color space, codec framing, or file format.

The codec/WAV and image/container boundaries are effects and error surfaces.
Backend tensor success is necessary but not sufficient to claim media output
validity. Exact format support is currently partial.
