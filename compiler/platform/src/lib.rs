#![forbid(unsafe_code)]

mod database;
mod model_graph;
mod tensor;

pub fn database_source() -> &'static str {
    database::source()
}

pub fn model_graph_source(rocm: bool) -> String {
    model_graph::source(rocm)
}

pub fn tensor_source(
    relu: bool,
    add: bool,
    matmul: bool,
    transpose: bool,
    scale: bool,
    softmax_rows: bool,
    layer_norm: bool,
    relu_backward: bool,
    softmax_backward: bool,
    layer_norm_backward: bool,
    autodiff: bool,
    rocm: bool,
) -> String {
    tensor::source(
        relu,
        add,
        matmul,
        transpose,
        scale,
        softmax_rows,
        layer_norm,
        relu_backward,
        softmax_backward,
        layer_norm_backward,
        autodiff,
        rocm,
    )
}
