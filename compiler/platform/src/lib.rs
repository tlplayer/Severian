#![forbid(unsafe_code)]

mod database;
mod tensor;

pub fn database_source() -> &'static str {
    database::source()
}

pub fn tensor_source(relu: bool, add: bool, matmul: bool) -> String {
    tensor::source(relu, add, matmul)
}
