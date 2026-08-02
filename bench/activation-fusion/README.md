# Automatic activation fusion

Both programs compute `Swish(FastTanh(Relu(X)))` over 262,144 values. The fused
source expresses the operations as one nested model expression. The compiler
combines it into one elementwise traversal automatically. The materialized
control assigns each activation result to a binding, forcing three traversals
and two intermediate tensors.

Neither program contains a fusion or hardware-placement request. Run the
correctness-gated comparison with:

```sh
python3 bench/activation-fusion/run.py --samples 15 --warmup 3
```
