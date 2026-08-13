"""Popcorn vectorsum_v2 protocol adapter.

`launch` is supplied by the generated Popcorn submission. It compiles and
invokes embedded Severian TTIR. This file only translates the benchmark's
tuple-shaped calling convention.
"""

from task import input_t, output_t


def _custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    return launch(input_tensor, output_tensor)


custom_kernel = _custom_kernel
