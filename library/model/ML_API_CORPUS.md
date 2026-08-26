# Machine-learning API corpus

Severian uses snake_case spellings and keeps external frameworks at the import
boundary. Names enter the public API only with typed behavior, compiler
lowering, and tests; empty compatibility stubs do not count as coverage.

## Model lifecycle

`load`, `save`, `fit`, `train`, `eval`, `predict`, `predict_proba`, `score`,
`evaluate`, `compile`, `forward`, `parameters`, `weights`, `state`,
`load_state`, and `summary`.

`model.load` acquires a frozen artifact independently of checkpoint identity.
Architecture/configuration lowering happens after acquisition. An existing
network's `load` or `load_state` loads parameters into its declared
architecture; `file.read` is not a synonym.

## Tensor operations

`tensor`, `zeros`, `ones`, `empty`, `full`, `rand`, `randn`, `arange`,
`linspace`, `shape`, `size`, `reshape`, `view`, `flatten`, `squeeze`,
`unsqueeze`, `transpose`, `permute`, `concat`, `stack`, `split`, `chunk`, `sum`,
`mean`, `min`, `max`, `argmin`, `argmax`, `std`, `var`, `matmul`, `dot`,
`einsum`, `where`, `clamp`, `clip`, `topk`, `sort`, `argsort`, `exp`, `log`,
`sqrt`, `abs`, `pow`, `softmax`, `log_softmax`, `sigmoid`, `relu`, `gelu`,
`silu`, and `tanh`.

## Neural-network layers

`linear`, `dense`, `embedding`, `conv1d`, `conv2d`, `conv3d`,
`conv_transpose2d`, `max_pool1d`, `max_pool2d`, `avg_pool1d`, `avg_pool2d`,
`adaptive_avg_pool2d`, `batch_norm`, `layer_norm`, `group_norm`, `dropout`,
`relu`, `gelu`, `silu`, `sigmoid`, `softmax`, `flatten`, `reshape`, `sequential`,
`attention`, `multihead_attention`, `rnn`, `lstm`, `gru`, `transformer`,
`transformer_encoder`, and `transformer_decoder`.

## Losses and optimization

`cross_entropy`, `binary_cross_entropy`, `binary_cross_entropy_logits`, `mse`,
`mae`, `huber`, `nll_loss`, `kl_divergence`, `cosine_embedding_loss`,
`triplet_loss`, `sgd`, `adam`, `adamw`, `adagrad`, `adadelta`, `rmsprop`,
`zero_grad`, `step`, `learning_rate`, `weight_decay`, `momentum`, `linear_lr`,
`cosine_lr`, `exponential_lr`, and `reduce_lr_on_plateau`.
