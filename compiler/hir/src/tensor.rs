pub const TENSOR_MATMUL_NATIVE_SYMBOL: &str = "__sev_tensor_matmul";

/// Backend-neutral identity for compiler-recognized tensor intrinsics.
///
/// The source name is intentionally absent: aliases and dtype-specific entry
/// points resolve to one semantic operation before MIR or backend lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorIntrinsic {
    Add,
    Subtract,
    Multiply,
    Divide,
    Matmul,
    Reshape,
    Transpose,
    Broadcast,
    BroadcastLike,
    Scale,
    AddScalar,
    Relu,
    Silu,
    Exp,
    Tanh,
    Rsqrt,
    Sigmoid,
    Gelu,
    Sum,
    SumLast,
    MeanLast,
    MaxLast,
    Softmax,
    SoftmaxAxis,
    Gather,
    Convert,
    ConvertLike,
    LayerNorm,
    DynamicSlice,
    DynamicUpdateSlice,
    DynamicUpdateSliceAxis,
    Slice,
    Cosine,
    Sine,
    Concatenate,
    Where,
}

impl TensorIntrinsic {
    /// Resolve an ABI symbol to tensor semantics. This is the sole mapping from
    /// runtime symbols to compiler operations.
    pub fn from_native_symbol(symbol: &str) -> Option<Self> {
        Some(match symbol {
            "__sev_tensor_add" => Self::Add,
            "__sev_tensor_subtract" | "__sev_tensor_f32_subtract" => Self::Subtract,
            "__sev_tensor_multiply"
            | "__sev_tensor_bf16_multiply"
            | "__sev_tensor_f32_multiply" => Self::Multiply,
            "__sev_tensor_divide" | "__sev_tensor_f32_divide" => Self::Divide,
            TENSOR_MATMUL_NATIVE_SYMBOL
            | "__sev_tensor_bf16_matmul"
            | "__sev_tensor_f32_matmul"
            | "__sev_tensor_f32_batched_matmul" => Self::Matmul,
            "__sev_tensor_reshape" => Self::Reshape,
            "__sev_tensor_transpose"
            | "__sev_tensor_bf16_transpose"
            | "__sev_tensor_f32_transpose" => Self::Transpose,
            "__sev_tensor_bf16_broadcast" | "__sev_tensor_f32_broadcast" => Self::Broadcast,
            "__sev_tensor_broadcast_like"
            | "__sev_tensor_bf16_broadcast_like"
            | "__sev_tensor_f32_broadcast_like" => Self::BroadcastLike,
            "__sev_tensor_scale" | "__sev_tensor_f32_scale" => Self::Scale,
            "__sev_tensor_add_scalar" | "__sev_tensor_f32_add_scalar" => Self::AddScalar,
            "__sev_tensor_relu" => Self::Relu,
            "__sev_tensor_silu" => Self::Silu,
            "__sev_tensor_exp" | "__sev_tensor_f32_exp" => Self::Exp,
            "__sev_tensor_tanh" => Self::Tanh,
            "__sev_tensor_rsqrt" | "__sev_tensor_f32_rsqrt" => Self::Rsqrt,
            "__sev_tensor_sigmoid" | "__sev_tensor_bf16_sigmoid" => Self::Sigmoid,
            "__sev_tensor_gelu" => Self::Gelu,
            "__sev_tensor_sum" => Self::Sum,
            "__sev_tensor_f32_sum_last" => Self::SumLast,
            "__sev_tensor_mean_last" | "__sev_tensor_f32_mean_last" => Self::MeanLast,
            "__sev_tensor_f32_max_last" => Self::MaxLast,
            "__sev_tensor_softmax_rows" => Self::Softmax,
            "__sev_tensor_f32_softmax_axis" => Self::SoftmaxAxis,
            "__sev_tensor_gather" | "__sev_tensor_bf16_gather" => Self::Gather,
            "__sev_tensor_to_f8e4m3fn"
            | "__sev_tensor_to_f8e5m2"
            | "__sev_tensor_to_f16"
            | "__sev_tensor_to_bf16"
            | "__sev_tensor_to_f32"
            | "__sev_tensor_to_f64"
            | "__sev_tensor_bf16_to_f32"
            | "__sev_tensor_f32_to_bf16" => Self::Convert,
            "__sev_tensor_convert_like" => Self::ConvertLike,
            "__sev_tensor_layer_norm" => Self::LayerNorm,
            "__sev_tensor_bf16_dynamic_slice" => Self::DynamicSlice,
            "__sev_tensor_bf16_dynamic_update_slice" => Self::DynamicUpdateSlice,
            "__sev_tensor_bf16_dynamic_update_slice_axis" => Self::DynamicUpdateSliceAxis,
            "__sev_tensor_slice" | "__sev_tensor_f32_slice" => Self::Slice,
            "__sev_tensor_f32_cosine" => Self::Cosine,
            "__sev_tensor_f32_sine" => Self::Sine,
            "__sev_tensor_f32_concatenate" => Self::Concatenate,
            "__sev_tensor_f32_where" => Self::Where,
            _ => return None,
        })
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::Matmul => "matmul",
            Self::Reshape => "reshape",
            Self::Transpose => "transpose",
            Self::Broadcast => "broadcast",
            Self::BroadcastLike => "broadcast_like",
            Self::Scale => "scale",
            Self::AddScalar => "add_scalar",
            Self::Relu => "relu",
            Self::Silu => "silu",
            Self::Exp => "exp",
            Self::Tanh => "tanh",
            Self::Rsqrt => "rsqrt",
            Self::Sigmoid => "sigmoid",
            Self::Gelu => "gelu",
            Self::Sum => "sum",
            Self::SumLast => "sum_last",
            Self::MeanLast => "mean_last",
            Self::MaxLast => "max_last",
            Self::Softmax => "softmax",
            Self::SoftmaxAxis => "softmax_axis",
            Self::Gather => "gather",
            Self::Convert => "convert",
            Self::ConvertLike => "convert_like",
            Self::LayerNorm => "layer_norm",
            Self::DynamicSlice => "dynamic_slice",
            Self::DynamicUpdateSlice => "dynamic_update_slice",
            Self::DynamicUpdateSliceAxis => "dynamic_update_slice_axis",
            Self::Slice => "slice",
            Self::Cosine => "cosine",
            Self::Sine => "sine",
            Self::Concatenate => "concatenate",
            Self::Where => "where",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_operations_have_one_dtype_polymorphic_symbol() {
        assert_eq!(
            TensorIntrinsic::from_native_symbol("__sev_tensor_add"),
            Some(TensorIntrinsic::Add)
        );
    }

    #[test]
    fn source_names_are_not_treated_as_intrinsic_identity() {
        assert_eq!(
            TensorIntrinsic::from_native_symbol("tensor.ranked_add"),
            None
        );
    }
}
