"""Popcorn vectorsum_v2 protocol adapter.

`launch` is supplied by the generated Severian kernel concatenated before this
adapter. Computation remains in that kernel; this file only translates the
benchmark's tuple-shaped calling convention.
"""

import torch
from task import input_t, output_t


def _custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    return launch(input_tensor, output_tensor)


custom_kernel = torch.compile(_custom_kernel, mode="reduce-overhead")
