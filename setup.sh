#!/usr/bin/env bash
set -euo pipefail

# Run from the Severian repository root.
test -d compiler || {
    echo "error: run from Severian repository root"
    exit 1
}

# ------------------------------------------------------------
# Purge directories that conflict with the golden architecture.
# ------------------------------------------------------------

# Source infrastructure is shared compiler infrastructure, not a frontend pass.
rm -rf compiler/frontend/source

# LLVM is reached through MLIR/backend lowering rather than being a parallel
# source-level lowering subsystem.
rm -rf compiler/transforms/lowering/llvm


# ------------------------------------------------------------
# Top-level compiler areas.
# ------------------------------------------------------------

mkdir -p \
    compiler/source/src \
    compiler/source/tests \
    compiler/diagnostics/src/{catalog,lint,render,tooling} \
    compiler/diagnostics/tests


# ------------------------------------------------------------
# Boundaries
# ------------------------------------------------------------

mkdir -p \
    compiler/boundaries/interface/src/{contract,model,symbol} \
    compiler/boundaries/interface/tests \
    compiler/boundaries/xxi/src/{model,resolve} \
    compiler/boundaries/xxi/tests \
    compiler/boundaries/ffi/src/{model,validate,marshal} \
    compiler/boundaries/ffi/tests \
    compiler/boundaries/abi/src/{model,registry,validate} \
    compiler/boundaries/abi/tests \
    compiler/boundaries/backend/src/{model,registry} \
    compiler/boundaries/backend/tests \
    compiler/boundaries/driver/src \
    compiler/boundaries/driver/tests


# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------

mkdir -p \
    compiler/frontend/lexer/src \
    compiler/frontend/lexer/tests \
    compiler/frontend/parser/src \
    compiler/frontend/parser/tests \
    compiler/frontend/ast/src \
    compiler/frontend/ast/tests \
    compiler/frontend/semantic/src/analyzer/{contracts,control,expression,generics} \
    compiler/frontend/semantic/src/{registry,resolve,types} \
    compiler/frontend/semantic/tests \
    compiler/frontend/hir/src \
    compiler/frontend/hir/tests \
    compiler/frontend/ownership/src \
    compiler/frontend/ownership/tests


# ------------------------------------------------------------
# Transforms
# ------------------------------------------------------------

mkdir -p \
    compiler/transforms/mir/src \
    compiler/transforms/mir/tests \
    compiler/transforms/mir/passes/src/{dataflow,loops} \
    compiler/transforms/mir/passes/tests \
    compiler/transforms/lowering/src/core/{bridge,control,expression,ffi,types} \
    compiler/transforms/lowering/tests \
    compiler/transforms/mlir/src \
    compiler/transforms/mlir/tests


# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------

mkdir -p \
    compiler/tools/project \
    compiler/tools/package \
    compiler/tools/build \
    compiler/tools/test \
    compiler/tools/developer


# ------------------------------------------------------------
# README contracts.
# ------------------------------------------------------------

touch \
    compiler/README.md \
    compiler/source/README.md \
    compiler/diagnostics/README.md \
    compiler/boundaries/README.md \
    compiler/boundaries/interface/README.md \
    compiler/boundaries/xxi/README.md \
    compiler/boundaries/ffi/README.md \
    compiler/boundaries/abi/README.md \
    compiler/boundaries/backend/README.md \
    compiler/boundaries/driver/README.md \
    compiler/frontend/README.md \
    compiler/frontend/lexer/README.md \
    compiler/frontend/parser/README.md \
    compiler/frontend/ast/README.md \
    compiler/frontend/semantic/README.md \
    compiler/frontend/semantic/src/analyzer/README.md \
    compiler/frontend/hir/README.md \
    compiler/frontend/ownership/README.md \
    compiler/transforms/README.md \
    compiler/transforms/mir/README.md \
    compiler/transforms/mir/passes/README.md \
    compiler/transforms/lowering/README.md \
    compiler/transforms/mlir/README.md \
    compiler/tools/README.md \
    compiler/tools/project/README.md \
    compiler/tools/package/README.md \
    compiler/tools/build/README.md \
    compiler/tools/test/README.md \
    compiler/tools/developer/README.md

tree compiler