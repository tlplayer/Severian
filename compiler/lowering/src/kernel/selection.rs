use super::{KernelBackend, KernelError, KernelIr, KernelSelection, KernelTarget};

pub fn select_backend(
    kernel: &KernelIr,
    requested: KernelBackend,
    target: KernelTarget,
) -> Result<KernelSelection, KernelError> {
    let (selected, reason, fallback) = match requested {
        KernelBackend::Auto => select_automatic(kernel, &target),
        KernelBackend::Triton => select_explicit_triton(kernel, &target)?,
        KernelBackend::Xla => (
            KernelBackend::Xla,
            "the portable StableHLO/XLA backend was explicitly requested".into(),
            None,
        ),
        KernelBackend::Llvm => {
            if target != KernelTarget::Cpu {
                return Err(unsupported(
                    kernel,
                    requested,
                    "the LLVM kernel path currently targets the CPU",
                ));
            }
            (
                KernelBackend::Llvm,
                "the native LLVM backend was explicitly requested".into(),
                None,
            )
        }
    };
    Ok(KernelSelection {
        requested,
        selected,
        target,
        reason,
        fallback,
    })
}

fn select_automatic(
    kernel: &KernelIr,
    target: &KernelTarget,
) -> (KernelBackend, String, Option<KernelBackend>) {
    match target {
        KernelTarget::Nvidia { .. } | KernelTarget::Amd { .. }
            if target.supports_triton() && kernel.triton_support().is_ok() =>
        {
            (
                KernelBackend::Triton,
                format!(
                    "recognized `{}` on {}; native Triton IR lowering is available",
                    kernel.operation.name(),
                    target.name()
                ),
                Some(KernelBackend::Xla),
            )
        }
        KernelTarget::Gpu | KernelTarget::Nvidia { .. } | KernelTarget::Amd { .. } => (
            KernelBackend::Xla,
            automatic_xla_reason(kernel, target),
            None,
        ),
        KernelTarget::Tpu => (
            KernelBackend::Xla,
            "TPU execution uses StableHLO and PJRT".into(),
            None,
        ),
        KernelTarget::Cpu => (
            KernelBackend::Llvm,
            "CPU execution uses the native LLVM path".into(),
            None,
        ),
    }
}

fn automatic_xla_reason(kernel: &KernelIr, target: &KernelTarget) -> String {
    if matches!(target, KernelTarget::Gpu) {
        return "GPU architecture is unspecified, so automatic policy keeps the portable StableHLO/XLA path; use a concrete target such as --target cuda:sm_90 or --target rocm:gfx1100 to enable specialized selection".into();
    }
    if matches!(
        target,
        KernelTarget::Nvidia { architecture: None } | KernelTarget::Amd { architecture: None }
    ) {
        return format!(
            "{} does not specify an architecture; using StableHLO/XLA until hardware compatibility is known",
            target.name()
        );
    }
    if !target.supports_triton() {
        return format!(
            "{} is outside the supported Triton hardware floor; using StableHLO/XLA",
            target.name()
        );
    }
    kernel.triton_support().err().map_or_else(
        || "no profitable direct Triton lowering was selected; using StableHLO/XLA".into(),
        |reason| format!("{reason}; using StableHLO/XLA"),
    )
}

fn select_explicit_triton(
    kernel: &KernelIr,
    target: &KernelTarget,
) -> Result<(KernelBackend, String, Option<KernelBackend>), KernelError> {
    if !target.is_gpu() {
        return Err(unsupported(
            kernel,
            KernelBackend::Triton,
            "Triton kernels require an NVIDIA or AMD GPU deployment target",
        ));
    }
    if target.is_known_triton_incompatible() {
        return Err(unsupported(
            kernel,
            KernelBackend::Triton,
            &format!(
                "{} is outside the supported Triton hardware floor",
                target.name()
            ),
        ));
    }
    kernel
        .triton_support()
        .map_err(|reason| unsupported(kernel, KernelBackend::Triton, &reason))?;
    let reason = if !target.supports_triton() {
        "Triton was explicitly requested without a concrete supported architecture; emitting portable TTIR with hardware validation deferred to deployment"
    } else {
        "the Triton backend was explicitly requested for supported GPU hardware"
    };
    Ok((
        KernelBackend::Triton,
        reason.into(),
        Some(KernelBackend::Xla),
    ))
}

fn unsupported(kernel: &KernelIr, backend: KernelBackend, reason: &str) -> KernelError {
    KernelError::UnsupportedBackend {
        kernel: kernel.name.clone(),
        backend,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_hir::{FunctionId, TensorDimension, TensorElementType, TensorType};
    use severian_mir::{LocalId, ReductionKind, ReductionOp, TensorOp, TensorOperand, ValueRef};

    fn reduction() -> KernelIr {
        let input =
            TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap();
        KernelIr {
            function: FunctionId::from_name("reduce"),
            name: "reduce".into(),
            parameters: vec![input],
            parameter_locals: vec![LocalId(0)],
            result: TensorType::ranked(TensorElementType::F32, &[]).unwrap(),
            operation: TensorOp::Reduction(ReductionOp {
                kind: ReductionKind::Sum,
                input: TensorOperand {
                    value: ValueRef {
                        id: None,
                        ty: Some(severian_hir::ValueType::Tensor(input)),
                        local: Some(LocalId(0)),
                        tensor_op: None,
                    },
                    ty: input,
                },
                axes: vec![0],
                axes_known: true,
                last_axis: false,
                result: TensorType::ranked(TensorElementType::F32, &[]).unwrap(),
                accumulation: TensorElementType::F32,
            }),
            policy: KernelBackend::Auto,
        }
    }

    #[test]
    fn automatic_selection_needs_a_supported_hardware_target() {
        let kernel = reduction();
        let generic = select_backend(&kernel, KernelBackend::Auto, KernelTarget::Gpu).unwrap();
        assert_eq!(generic.selected, KernelBackend::Xla);

        let supported = select_backend(
            &kernel,
            KernelBackend::Auto,
            KernelTarget::Nvidia {
                architecture: Some("sm_90".into()),
            },
        )
        .unwrap();
        assert_eq!(supported.selected, KernelBackend::Triton);
        assert_eq!(supported.fallback, Some(KernelBackend::Xla));

        let old = select_backend(
            &kernel,
            KernelBackend::Auto,
            KernelTarget::Nvidia {
                architecture: Some("sm_75".into()),
            },
        )
        .unwrap();
        assert_eq!(old.selected, KernelBackend::Xla);
    }
}
