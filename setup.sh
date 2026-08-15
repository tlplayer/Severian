#!/usr/bin/env bash
set -euo pipefail

# Run from Severian repository root.

dirs=(
    # Compiler analysis / semantic concepts
    compiler/ownership/src/analysis
    compiler/semantic/src/analysis

    # Generic optimization passes
    compiler/passes/src/canonicalize
    compiler/passes/src/control_flow
    compiler/passes/src/dataflow
    compiler/passes/src/inlining
    compiler/passes/src/loops

    # XLA-inspired tensor/compiler passes
    compiler/passes/src/xla
    compiler/passes/src/xla/algebraic
    compiler/passes/src/xla/fusion
    compiler/passes/src/xla/layout
    compiler/passes/src/xla/memory
    compiler/passes/src/xla/scheduling

    # IREE-inspired lowering/codegen structure
    compiler/passes/src/iree
    compiler/passes/src/iree/dispatch
    compiler/passes/src/iree/tiling
    compiler/passes/src/iree/vectorization
    compiler/passes/src/iree/bufferization

    # StableHLO / XLA integ
    compiler/xla/src
    compiler/xla/src/stablehlo
    compiler/xla/src/pjrt

    # Lowering
    compiler/lowering/src/tensor
    compiler/lowering/src/stablehlo
    compiler/lowering/src/llvm
    compiler/lowering/src/gpu

    runtime/src
    runtime/src/scheduler
    runtime/src/sync
    runtime/src/channel
    runtime/src/netpoll

    # Platform abstraction
    compiler/platform/src
    compiler/platform/src/cpu
    compiler/platform/src/gpu

    # Tests
    tests/compiler/xla
    tests/compiler/passes
    tests/runtime
)

files=(
    # Ownership compiler concepts
    compiler/ownership/src/analysis/borrow_check.rs
    compiler/ownership/src/analysis/move_check.rs
    compiler/ownership/src/analysis/escape.rs
    compiler/ownership/src/analysis/liveness.rs
    compiler/ownership/src/analysis/alias.rs

    # Semantic analysis
    compiler/semantic/src/analysis/effects.rs
    compiler/semantic/src/analysis/types.rs
    compiler/semantic/src/analysis/traits.rs

    # Generic compiler optimization
    compiler/passes/src/canonicalize/mod.rs
    compiler/passes/src/control_flow/mod.rs
    compiler/passes/src/dataflow/mod.rs
    compiler/passes/src/inlining/mod.rs
    compiler/passes/src/loops/mod.rs

    # XLA concepts
    compiler/passes/src/xla/mod.rs
    compiler/passes/src/xla/algebraic/mod.rs
    compiler/passes/src/xla/algebraic/simplify.rs
    compiler/passes/src/xla/algebraic/constant_fold.rs

    compiler/passes/src/xla/fusion/mod.rs
    compiler/passes/src/xla/fusion/instruction_fusion.rs
    compiler/passes/src/xla/fusion/loop_fusion.rs
    compiler/passes/src/xla/fusion/multi_output_fusion.rs

    compiler/passes/src/xla/layout/mod.rs
    compiler/passes/src/xla/layout/assignment.rs
    compiler/passes/src/xla/layout/normalization.rs

    compiler/passes/src/xla/memory/mod.rs
    compiler/passes/src/xla/memory/buffer_assignment.rs
    compiler/passes/src/xla/memory/buffer_reuse.rs

    compiler/passes/src/xla/scheduling/mod.rs
    compiler/passes/src/xla/scheduling/instruction_scheduler.rs

    # IREE concepts
    compiler/passes/src/iree/mod.rs

    compiler/passes/src/iree/dispatch/mod.rs
    compiler/passes/src/iree/dispatch/formation.rs
    compiler/passes/src/iree/dispatch/fusion.rs

    compiler/passes/src/iree/tiling/mod.rs
    compiler/passes/src/iree/tiling/tile.rs

    compiler/passes/src/iree/vectorization/mod.rs
    compiler/passes/src/iree/vectorization/vectorize.rs

    compiler/passes/src/iree/bufferization/mod.rs
    compiler/passes/src/iree/bufferization/bufferize.rs

    # XLA integ crate
    compiler/xla/Cargo.toml
    compiler/xla/src/lib.rs
    compiler/xla/src/pipeline.rs
    compiler/xla/src/client.rs

    compiler/xla/src/stablehlo/mod.rs
    compiler/xla/src/stablehlo/export.rs
    compiler/xla/src/stablehlo/import.rs
    compiler/xla/src/stablehlo/types.rs

    compiler/xla/src/pjrt/mod.rs
    compiler/xla/src/pjrt/client.rs
    compiler/xla/src/pjrt/device.rs
    compiler/xla/src/pjrt/executable.rs
    compiler/xla/src/pjrt/buffer.rs

    # Lowering
    compiler/lowering/src/tensor/mod.rs
    compiler/lowering/src/tensor/linalg.rs

    compiler/lowering/src/stablehlo/mod.rs
    compiler/lowering/src/stablehlo/ops.rs

    compiler/lowering/src/llvm/mod.rs
    compiler/lowering/src/gpu/mod.rs

    # Runtime crate — Go runtime concepts
    runtime/Cargo.toml
    runtime/src/lib.rs

    runtime/src/scheduler/mod.rs
    runtime/src/scheduler/task.rs
    runtime/src/scheduler/queue.rs
    runtime/src/scheduler/worker.rs
    runtime/src/scheduler/park.rs
    runtime/src/scheduler/wake.rs

    runtime/src/channel/mod.rs
    runtime/src/channel/channel.rs
    runtime/src/channel/select.rs

    runtime/src/sync/mod.rs
    runtime/src/sync/mutex.rs
    runtime/src/sync/rwlock.rs
    runtime/src/sync/semaphore.rs
    runtime/src/sync/atomic.rs
    runtime/src/sync/once.rs

    runtime/src/netpoll/mod.rs
    runtime/src/netpoll/poller.rs

    runtime/src/task.rs
    runtime/src/thread.rs
    runtime/src/time.rs

    # Platform
    compiler/platform/Cargo.toml
    compiler/platform/src/lib.rs
    compiler/platform/src/cpu/mod.rs
    compiler/platform/src/gpu/mod.rs

    # Test placeholders
    tests/compiler/xla/stablehlo.sev
    tests/compiler/xla/fusion.sev
    tests/compiler/xla/algebraic.sev

    tests/compiler/passes/inlining.sev
    tests/compiler/passes/escape.sev
    tests/compiler/passes/liveness.sev

    tests/runtime/channel.sev
    tests/runtime/select.sev
    tests/runtime/scheduler.sev
    tests/runtime/mutex.sev
)

for dir in "${dirs[@]}"; do
    mkdir -p "$dir"
done

for file in "${files[@]}"; do
    mkdir -p "$(dirname "$file")"
    touch "$file"
done

echo "Created ${#dirs[@]} directories and ${#files[@]} placeholder files."