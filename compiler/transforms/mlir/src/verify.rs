use crate::emit::{artifact_symbol, MlirArtifact, MlirError};
use crate::ffi;
use crate::library::registered_libraries;
use severian_artifact::ArtifactId;
use severian_lir::{LoweredFloatFormat, LoweredType};
use severian_target::TargetSpec;
use std::collections::BTreeSet;
use std::ffi::c_void;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedMlirArtifact {
    id: ArtifactId,
    module: String,
    target: String,
    gpu_architecture: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuLaunchArtifact {
    pub id: ArtifactId,
    pub inputs: Vec<LoweredType>,
    pub outputs: Vec<LoweredType>,
}

impl VerifiedMlirArtifact {
    pub const fn id(&self) -> ArtifactId {
        self.id
    }
}

pub fn verify_artifact(
    id: ArtifactId,
    artifact: MlirArtifact,
    target: &TargetSpec,
) -> Result<VerifiedMlirArtifact, MlirError> {
    let context = Context::new();
    let module = Module::parse(&context, &artifact.module, "artifact").map_err(|error| {
        MlirError::ParseFailed(format!(
            "{error}; generated artifact:\n{}",
            numbered_excerpt(&artifact.module, 80)
        ))
    })?;
    let entry = module.sole_entry()?;
    if !module.operation_has_body(entry) {
        return Err(MlirError::EntryFunctionIsDeclaration);
    }
    module.verify("artifact module")?;
    module.verify_allowed_dialects(target)?;
    verify_entry_signature(&context, entry, &artifact.inputs, &artifact.outputs)?;
    let gpu_architecture =
        operation_string_attribute(module.operation(), "severian.gpu.architecture");
    Ok(VerifiedMlirArtifact {
        id,
        module: module.print(),
        target: target.triple.clone(),
        gpu_architecture,
    })
}

fn numbered_excerpt(module: &str, limit: usize) -> String {
    module
        .lines()
        .take(limit)
        .enumerate()
        .map(|(line, text)| format!("{:>4}: {text}", line + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn compose(
    normal: &str,
    artifacts: &[VerifiedMlirArtifact],
    target: &TargetSpec,
) -> Result<String, MlirError> {
    let context = Context::new();
    let module = Module::parse(&context, normal, "ordinary module").map_err(|error| {
        MlirError::ParseFailed(format!(
            "{error}; generated ordinary module:\n{}",
            numbered_excerpt(normal, 800)
        ))
    })?;
    module.verify("ordinary module")?;
    module.verify_allowed_dialects(target)?;

    let symbol_table = SymbolTable::new(&module)?;
    let mut artifact_ids = BTreeSet::new();
    let mut composed_declarations = BTreeSet::new();
    let gpu_architectures = artifacts
        .iter()
        .filter_map(|artifact| artifact.gpu_architecture.as_deref())
        .collect::<BTreeSet<_>>();
    if gpu_architectures.len() > 1 {
        return Err(MlirError::TargetMismatch {
            artifact: gpu_architectures.into_iter().collect::<Vec<_>>().join(","),
            composition: target.triple.clone(),
        });
    }
    if let Some(architecture) = gpu_architectures.into_iter().next() {
        let value = unsafe { ffi::mlirStringAttrGet(context.raw, ffi::string_ref(architecture)) };
        unsafe {
            ffi::mlirOperationSetAttributeByName(
                module.operation(),
                ffi::string_ref("severian.gpu.architecture"),
                value,
            )
        };
    }
    for artifact in artifacts {
        if artifact.target != target.triple {
            return Err(MlirError::TargetMismatch {
                artifact: artifact.target.clone(),
                composition: target.triple.clone(),
            });
        }
        let generated = Module::parse(&context, &artifact.module, "verified artifact")?;
        let entry = generated.sole_entry()?;
        if !generated.operation_has_body(entry) {
            return Err(MlirError::EntryFunctionIsDeclaration);
        }
        if !artifact_ids.insert(artifact.id) {
            return Err(MlirError::DuplicateSymbol(artifact_symbol(artifact.id)));
        }
        let symbol = artifact_symbol(artifact.id);
        if let Some(declaration) = symbol_table.lookup(&symbol) {
            if operation_name(declaration) != "func.func" || module.operation_has_body(declaration)
            {
                return Err(MlirError::DuplicateSymbol(symbol));
            }
            unsafe { ffi::mlirSymbolTableErase(symbol_table.raw, declaration) };
        }
        let mut declaration = unsafe { ffi::mlirBlockGetFirstOperation(generated.body()) };
        while !declaration.is_null() {
            if declaration.ptr != entry.ptr && operation_name(declaration) == "func.func" {
                if let Some(name) = operation_symbol_name(declaration) {
                    if composed_declarations.insert(name.clone())
                        && symbol_table.lookup(&name).is_none()
                    {
                        let cloned = unsafe { ffi::mlirOperationClone(declaration) };
                        unsafe { ffi::mlirBlockAppendOwnedOperation(module.body(), cloned) };
                    }
                }
            }
            declaration = unsafe { ffi::mlirOperationGetNextInBlock(declaration) };
        }
        let cloned = unsafe { ffi::mlirOperationClone(entry) };
        let symbol_name = unsafe { ffi::mlirStringAttrGet(context.raw, ffi::string_ref(&symbol)) };
        unsafe {
            ffi::mlirOperationSetAttributeByName(cloned, ffi::string_ref("sym_name"), symbol_name);
            ffi::mlirBlockAppendOwnedOperation(module.body(), cloned);
        }
    }

    compose_registered_libraries(&context, &module, target)?;

    if module.verify("composed module").is_err() {
        let printed = module.print();
        let excerpt = mismatched_call_signatures(&printed).join("\n");
        return Err(MlirError::VerificationFailed(format!(
            "composed module; mismatched calls:\n{excerpt}"
        )));
    }
    module.verify_allowed_dialects(target)?;
    Ok(module.print())
}

fn compose_registered_libraries(
    context: &Context,
    module: &Module<'_>,
    target: &TargetSpec,
) -> Result<(), MlirError> {
    for library in registered_libraries() {
        let symbols = SymbolTable::new(module)?;
        let required = library
            .exports
            .iter()
            .filter(|symbol| symbols.lookup(symbol).is_some())
            .copied()
            .collect::<BTreeSet<_>>();
        if required.is_empty() {
            continue;
        }
        if library
            .pointer_bits
            .is_some_and(|bits| bits != target.pointer_bits())
        {
            return Err(MlirError::UnsupportedOperation(format!(
                "MLIR library {} v{} requires a {}-bit pointer ABI, target {} uses {} bits",
                library.id,
                library.abi_version,
                library.pointer_bits.expect("checked as present"),
                target.triple,
                target.pointer_bits()
            )));
        }
        for symbol in &required {
            let declaration = symbols
                .lookup(symbol)
                .expect("required library symbol was discovered in this table");
            if operation_name(declaration) != "func.func" || module.operation_has_body(declaration)
            {
                return Err(MlirError::DuplicateSymbol((*symbol).to_owned()));
            }
        }
        let source = Module::parse(
            context,
            library.module,
            &format!("MLIR library {} v{}", library.id, library.abi_version),
        )?;
        source.verify(&format!("MLIR library {}", library.id))?;
        source.verify_allowed_dialects(target)?;
        let declared_id = operation_string_attribute(source.operation(), "severian.library_id");
        let declared_version =
            operation_integer_attribute(source.operation(), "severian.abi_version");
        if declared_id.as_deref() != Some(library.id)
            || declared_version != Some(i64::from(library.abi_version))
        {
            return Err(MlirError::VerificationFailed(format!(
                "MLIR library registry expects {} v{}, module declares {:?} v{:?}",
                library.id, library.abi_version, declared_id, declared_version
            )));
        }
        let source_symbols = SymbolTable::new(&source)?;
        for symbol in &required {
            if source_symbols
                .lookup(symbol)
                .is_none_or(|definition| !source.operation_has_body(definition))
            {
                return Err(MlirError::VerificationFailed(format!(
                    "MLIR library {} does not define required export `{symbol}`",
                    library.id
                )));
            }
        }
        let mut operation = unsafe { ffi::mlirBlockGetFirstOperation(source.body()) };
        while !operation.is_null() {
            let next = unsafe { ffi::mlirOperationGetNextInBlock(operation) };
            if let Some(name) = operation_symbol_name(operation) {
                let import = required.contains(name.as_str())
                    || library.dependencies.contains(&name.as_str());
                if import {
                    if let Some(existing) = symbols.lookup(&name) {
                        if required.contains(name.as_str()) {
                            unsafe { ffi::mlirSymbolTableErase(symbols.raw, existing) };
                        }
                    }
                    if symbols.lookup(&name).is_none() {
                        let cloned = unsafe { ffi::mlirOperationClone(operation) };
                        unsafe { ffi::mlirBlockAppendOwnedOperation(module.body(), cloned) };
                    }
                }
            }
            operation = next;
        }
        module.verify(&format!("module composed with MLIR library {}", library.id))?;
    }
    Ok(())
}

fn mismatched_call_signatures(module: &str) -> Vec<String> {
    let mut definitions = std::collections::BTreeMap::new();
    for line in module.lines() {
        let Some(signature) = line
            .split("function_type = ")
            .nth(1)
            .and_then(|tail| tail.split(", sym_name = \"").next())
        else {
            continue;
        };
        let Some(symbol) = line
            .split(", sym_name = \"")
            .nth(1)
            .and_then(|tail| tail.split('"').next())
        else {
            continue;
        };
        let inputs = signature.split(" -> ").next().unwrap_or(signature);
        definitions.insert(symbol.to_owned(), inputs.to_owned());
    }
    module
        .lines()
        .filter_map(|line| {
            let tail = line.split("<{callee = @").nth(1)?;
            let symbol = tail.split("}>").next()?;
            let provided = line.split("}> : ").nth(1)?.split(" -> ").next()?;
            let expected = definitions.get(symbol)?;
            (provided != expected).then(|| {
                format!(
                    "@{symbol}: expected {expected}, provided {provided}; operation: {}",
                    line.trim()
                )
            })
        })
        .take(40)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_string_library_replaces_requested_declarations() {
        let ordinary = r#"
module {
  func.func private @__sev_string_concat(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  func.func private @__sev_string_compare(!llvm.ptr, !llvm.ptr) -> i32
  func.func private @__sev_string_release(!llvm.ptr)
  func.func @entry(%left: !llvm.ptr, %right: !llvm.ptr) -> i32 {
    %joined = func.call @__sev_string_concat(%left, %right) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %comparison = func.call @__sev_string_compare(%joined, %right) : (!llvm.ptr, !llvm.ptr) -> i32
    func.call @__sev_string_release(%joined) : (!llvm.ptr) -> ()
    return %comparison : i32
  }
}
"#;

        let composed = compose(ordinary, &[], &TargetSpec::host()).unwrap();

        assert!(composed.contains("func.func @__sev_string_concat("));
        assert!(composed.contains("func.func @__sev_string_compare("));
        assert!(composed.contains("func.func @__sev_string_release("));
        assert!(!composed.contains("func.func private @__sev_string_concat"));
        assert!(composed.contains("func.func private @malloc"));
        assert!(composed.contains("func.func private @free"));
    }

    #[test]
    fn unused_registered_library_is_not_imported() {
        let ordinary = "module { func.func @entry() { return } }";
        let composed = compose(ordinary, &[], &TargetSpec::host()).unwrap();

        assert!(!composed.contains("__sev_string_concat"));
        assert!(!composed.contains("severian.library_id"));
    }

    #[test]
    fn a_string_export_definition_cannot_be_silently_replaced() {
        let ordinary = r#"
module {
  func.func @__sev_string_compare(%left: !llvm.ptr, %right: !llvm.ptr) -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
}
"#;

        assert!(matches!(
            compose(ordinary, &[], &TargetSpec::host()),
            Err(MlirError::DuplicateSymbol(symbol)) if symbol == "__sev_string_compare"
        ));
    }

    #[test]
    fn legacy_string_layout_rejects_a_32_bit_target_before_import() {
        let ordinary =
            "module { func.func private @__sev_string_concat(!llvm.ptr, !llvm.ptr) -> !llvm.ptr }";

        assert!(matches!(
            compose(ordinary, &[], &TargetSpec::new("x86-unknown-linux")),
            Err(MlirError::UnsupportedOperation(message))
                if message.contains("requires a 64-bit pointer ABI")
        ));
    }
}

/// Replaces compiled-region declarations with host functions that call the
/// Severian GPU launcher ABI. Kernel graphs and binaries remain outside this
/// host MLIR module; only their stable artifact identity crosses this boundary.
pub fn compose_gpu_launchers(
    normal: &str,
    launchers: &[GpuLaunchArtifact],
    target: &TargetSpec,
) -> Result<String, MlirError> {
    if launchers.is_empty() {
        return Ok(normal.to_owned());
    }
    let context = Context::new();
    let module = Module::parse(&context, normal, "ordinary module").map_err(|error| {
        MlirError::ParseFailed(format!(
            "{error}; generated ordinary module:\n{}",
            numbered_excerpt(normal, 800)
        ))
    })?;
    module.verify("ordinary module")?;
    module.verify_allowed_dialects(target)?;
    let symbol_table = SymbolTable::new(&module)?;
    let mut artifact_ids = BTreeSet::new();

    for launcher in launchers {
        if !artifact_ids.insert(launcher.id) {
            return Err(MlirError::DuplicateSymbol(artifact_symbol(launcher.id)));
        }
        let artifact = artifact_symbol(launcher.id);
        let Some(declaration) = symbol_table.lookup(&artifact) else {
            return Err(MlirError::DuplicateSymbol(artifact));
        };
        if operation_name(declaration) != "func.func" || module.operation_has_body(declaration) {
            return Err(MlirError::DuplicateSymbol(artifact));
        }
        verify_entry_signature(&context, declaration, &launcher.inputs, &launcher.outputs)?;

        let generated_source = gpu_launcher_module(launcher)?;
        let generated = Module::parse(&context, &generated_source, "GPU launcher module")?;
        generated.verify("GPU launcher module")?;
        let runtime_symbol = gpu_launcher_symbol(launcher.id);
        let runtime_declaration = symbol_table.lookup(&runtime_symbol);

        let mut current = unsafe { ffi::mlirBlockGetFirstOperation(generated.body()) };
        while !current.is_null() {
            let next = unsafe { ffi::mlirOperationGetNextInBlock(current) };
            let name = operation_symbol_name(current);
            if name.as_deref() == Some("entry") {
                let cloned = unsafe { ffi::mlirOperationClone(current) };
                let symbol_name =
                    unsafe { ffi::mlirStringAttrGet(context.raw, ffi::string_ref(&artifact)) };
                unsafe {
                    ffi::mlirOperationSetAttributeByName(
                        cloned,
                        ffi::string_ref("sym_name"),
                        symbol_name,
                    );
                    ffi::mlirBlockAppendOwnedOperation(module.body(), cloned);
                }
            } else if name.as_deref() == Some(runtime_symbol.as_str())
                && runtime_declaration.is_none()
            {
                let cloned = unsafe { ffi::mlirOperationClone(current) };
                unsafe { ffi::mlirBlockAppendOwnedOperation(module.body(), cloned) };
            }
            current = next;
        }
        unsafe { ffi::mlirSymbolTableErase(symbol_table.raw, declaration) };
    }

    module.verify("module with GPU launchers")?;
    module.verify_allowed_dialects(target)?;
    Ok(module.print())
}

fn gpu_launcher_module(launcher: &GpuLaunchArtifact) -> Result<String, MlirError> {
    let input_types = launcher
        .inputs
        .iter()
        .map(crate::emit::mlir_type)
        .collect::<Result<Vec<_>, _>>()?;
    let output_types = launcher
        .outputs
        .iter()
        .map(crate::emit::mlir_type)
        .collect::<Result<Vec<_>, _>>()?;
    let parameters = input_types
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("%arg{index}: {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    let arguments = (0..input_types.len())
        .map(|index| format!("%arg{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let input_signature = input_types.join(", ");
    let result_signature = match output_types.as_slice() {
        [] => String::new(),
        [output] => format!(" -> {output}"),
        outputs => format!(" -> ({})", outputs.join(", ")),
    };
    let (assignment, return_operation) = match output_types.as_slice() {
        [] => (String::new(), "    return\n".to_owned()),
        outputs => {
            let values = (0..outputs.len())
                .map(|index| format!("%result{index}"))
                .collect::<Vec<_>>()
                .join(", ");
            (
                format!("{values} = "),
                format!("    return {values} : {}\n", outputs.join(", ")),
            )
        }
    };
    let runtime = gpu_launcher_symbol(launcher.id);
    Ok(format!(
        "module {{\n  func.func private @{runtime}({input_signature}){result_signature}\n  func.func private @entry({parameters}){result_signature} {{\n    {assignment}func.call @{runtime}({arguments}) : ({input_signature}){result_signature}\n{return_operation}  }}\n}}"
    ))
}

fn gpu_launcher_symbol(artifact: ArtifactId) -> String {
    format!("__sev_gpu_launch_{}", artifact.index())
}

struct Context {
    raw: ffi::MlirContext,
}

impl Context {
    fn new() -> Self {
        unsafe {
            let registry = ffi::mlirDialectRegistryCreate();
            ffi::mlirRegisterAllDialects(registry);
            for dialect in [
                ffi::mlirGetDialectHandle__arith__(),
                ffi::mlirGetDialectHandle__async__(),
                ffi::mlirGetDialectHandle__cf__(),
                ffi::mlirGetDialectHandle__func__(),
                ffi::mlirGetDialectHandle__gpu__(),
                ffi::mlirGetDialectHandle__llvm__(),
                ffi::mlirGetDialectHandle__linalg__(),
                ffi::mlirGetDialectHandle__math__(),
                ffi::mlirGetDialectHandle__rocdl__(),
                ffi::mlirGetDialectHandle__scf__(),
                ffi::mlirGetDialectHandle__tensor__(),
                ffi::mlirGetDialectHandle__vector__(),
            ] {
                ffi::mlirDialectHandleInsertDialect(dialect, registry);
            }
            let raw = ffi::mlirContextCreate();
            ffi::mlirContextAppendDialectRegistry(raw, registry);
            ffi::mlirContextSetAllowUnregisteredDialects(raw, false);
            ffi::mlirDialectRegistryDestroy(registry);
            Self { raw }
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::mlirContextDestroy(self.raw) }
    }
}

struct Module<'context> {
    raw: ffi::MlirModule,
    _context: &'context Context,
}

impl<'context> Module<'context> {
    fn parse(
        context: &'context Context,
        source: &str,
        description: &str,
    ) -> Result<Self, MlirError> {
        let raw = unsafe { ffi::mlirModuleCreateParse(context.raw, ffi::string_ref(source)) };
        if raw.is_null() {
            return Err(MlirError::ParseFailed(description.to_owned()));
        }
        Ok(Self {
            raw,
            _context: context,
        })
    }

    fn operation(&self) -> ffi::MlirOperation {
        unsafe { ffi::mlirModuleGetOperation(self.raw) }
    }

    fn body(&self) -> ffi::MlirBlock {
        unsafe { ffi::mlirModuleGetBody(self.raw) }
    }

    fn sole_entry(&self) -> Result<ffi::MlirOperation, MlirError> {
        let mut current = unsafe { ffi::mlirBlockGetFirstOperation(self.body()) };
        let mut entry = None;
        let mut count = 0usize;
        let mut declarations = 0usize;
        while !current.is_null() {
            if operation_name(current) == "func.func" {
                if self.operation_has_body(current) {
                    if operation_string_attribute(current, "sym_visibility").as_deref()
                        != Some("private")
                    {
                        count += 1;
                        entry = Some(current);
                    }
                } else {
                    declarations += 1;
                }
            }
            current = unsafe { ffi::mlirOperationGetNextInBlock(current) };
        }
        if count == 1 {
            Ok(entry.expect("one entry was counted"))
        } else if count == 0 && declarations == 1 {
            Err(MlirError::EntryFunctionIsDeclaration)
        } else {
            Err(MlirError::EntryFunctionCount(count))
        }
    }

    fn operation_has_body(&self, operation: ffi::MlirOperation) -> bool {
        let region = unsafe { ffi::mlirOperationGetFirstRegion(operation) };
        !region.is_null() && !unsafe { ffi::mlirRegionGetFirstBlock(region) }.is_null()
    }

    fn verify(&self, description: &str) -> Result<(), MlirError> {
        if unsafe { ffi::mlirOperationVerify(self.operation()) } {
            Ok(())
        } else {
            Err(MlirError::VerificationFailed(description.to_owned()))
        }
    }

    fn verify_allowed_dialects(&self, target: &TargetSpec) -> Result<(), MlirError> {
        let mut allowed = [
            "builtin",
            "arith",
            "async",
            "bufferization",
            "cf",
            "func",
            "linalg",
            "llvm",
            "math",
            "scf",
            "tensor",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        for capability in target.capabilities.iter() {
            if let Some(dialect) = capability.strip_prefix("mlir.dialect.") {
                allowed.insert(dialect.to_owned());
            }
        }
        let mut operations = vec![self.operation()];
        while let Some(operation) = operations.pop() {
            let name = operation_name(operation);
            let dialect = name
                .split_once('.')
                .map_or("builtin", |(dialect, _)| dialect);
            if !allowed.contains(dialect) {
                return Err(MlirError::DialectNotAllowed {
                    dialect: dialect.to_owned(),
                    target: target.triple.clone(),
                });
            }
            collect_children(operation, &mut operations);
        }
        Ok(())
    }

    fn print(&self) -> String {
        let mut output = String::new();
        unsafe {
            ffi::mlirOperationPrint(
                self.operation(),
                append_printed_text,
                (&mut output as *mut String).cast(),
            );
        }
        output
    }
}

fn operation_symbol_name(operation: ffi::MlirOperation) -> Option<String> {
    operation_string_attribute(operation, "sym_name")
}

fn operation_string_attribute(operation: ffi::MlirOperation, name: &str) -> Option<String> {
    let attribute =
        unsafe { ffi::mlirOperationGetAttributeByName(operation, ffi::string_ref(name)) };
    if attribute.is_null() {
        return None;
    }
    let value = unsafe { ffi::mlirStringAttrGetValue(attribute) };
    if value.data.is_null() {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(value.data.cast::<u8>(), value.length) };
    Some(String::from_utf8_lossy(bytes).into_owned())
}

fn operation_integer_attribute(operation: ffi::MlirOperation, name: &str) -> Option<i64> {
    let attribute =
        unsafe { ffi::mlirOperationGetAttributeByName(operation, ffi::string_ref(name)) };
    (!attribute.is_null()).then(|| unsafe { ffi::mlirIntegerAttrGetValueInt(attribute) })
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe { ffi::mlirModuleDestroy(self.raw) }
    }
}

struct SymbolTable {
    raw: ffi::MlirSymbolTable,
}

impl SymbolTable {
    fn new(module: &Module<'_>) -> Result<Self, MlirError> {
        let raw = unsafe { ffi::mlirSymbolTableCreate(module.operation()) };
        if raw.is_null() {
            Err(MlirError::MlirApi(
                "builtin module does not expose a symbol table".into(),
            ))
        } else {
            Ok(Self { raw })
        }
    }

    fn lookup(&self, name: &str) -> Option<ffi::MlirOperation> {
        let operation = unsafe { ffi::mlirSymbolTableLookup(self.raw, ffi::string_ref(name)) };
        (!operation.is_null()).then_some(operation)
    }
}

impl Drop for SymbolTable {
    fn drop(&mut self) {
        unsafe { ffi::mlirSymbolTableDestroy(self.raw) }
    }
}

fn verify_entry_signature(
    context: &Context,
    entry: ffi::MlirOperation,
    inputs: &[LoweredType],
    outputs: &[LoweredType],
) -> Result<(), MlirError> {
    let attribute =
        unsafe { ffi::mlirOperationGetAttributeByName(entry, ffi::string_ref("function_type")) };
    if attribute.is_null() || !unsafe { ffi::mlirAttributeIsAType(attribute) } {
        return Err(MlirError::SignatureMismatch);
    }
    let function_type = unsafe { ffi::mlirTypeAttrGetValue(attribute) };
    if function_type.is_null() || !unsafe { ffi::mlirTypeIsAFunction(function_type) } {
        return Err(MlirError::SignatureMismatch);
    }
    if unsafe { ffi::mlirFunctionTypeGetNumInputs(function_type) } != inputs.len() as isize
        || unsafe { ffi::mlirFunctionTypeGetNumResults(function_type) } != outputs.len() as isize
    {
        return Err(MlirError::SignatureMismatch);
    }
    for (index, expected) in inputs.iter().enumerate() {
        let actual = unsafe { ffi::mlirFunctionTypeGetInput(function_type, index as isize) };
        if !unsafe { ffi::mlirTypeEqual(actual, lowered_type(context, expected)?) } {
            return Err(MlirError::SignatureMismatch);
        }
    }
    for (index, expected) in outputs.iter().enumerate() {
        let actual = unsafe { ffi::mlirFunctionTypeGetResult(function_type, index as isize) };
        if !unsafe { ffi::mlirTypeEqual(actual, lowered_type(context, expected)?) } {
            return Err(MlirError::SignatureMismatch);
        }
    }
    Ok(())
}

fn lowered_type(context: &Context, ty: &LoweredType) -> Result<ffi::MlirType, MlirError> {
    Ok(unsafe {
        match ty {
            LoweredType::Integer { bits, .. } => {
                ffi::mlirIntegerTypeGet(context.raw, (*bits).into())
            }
            LoweredType::Float {
                format: LoweredFloatFormat::Float8E4M3Fn,
            }
            | LoweredType::Float {
                format: LoweredFloatFormat::Float8E5M2,
            } => ffi::mlirIntegerTypeGet(context.raw, 8),
            LoweredType::Float {
                format: LoweredFloatFormat::Ieee(16),
            } => ffi::mlirF16TypeGet(context.raw),
            LoweredType::Float {
                format: LoweredFloatFormat::Ieee(32),
            } => ffi::mlirF32TypeGet(context.raw),
            LoweredType::Float {
                format: LoweredFloatFormat::Ieee(64),
            } => ffi::mlirF64TypeGet(context.raw),
            LoweredType::Float {
                format: LoweredFloatFormat::Ieee(80),
            } => ffi::mlirTypeParseGet(context.raw, ffi::string_ref("f80")),
            LoweredType::Float {
                format: LoweredFloatFormat::Ieee(128),
            } => ffi::mlirTypeParseGet(context.raw, ffi::string_ref("f128")),
            LoweredType::Float {
                format: LoweredFloatFormat::BrainFloat16,
            } => ffi::mlirBF16TypeGet(context.raw),
            LoweredType::Boolean => ffi::mlirIntegerTypeGet(context.raw, 1),
            LoweredType::String | LoweredType::Bytes => {
                ffi::mlirTypeParseGet(context.raw, ffi::string_ref("!llvm.ptr"))
            }
            LoweredType::None | LoweredType::Unit => ffi::mlirIntegerTypeGet(context.raw, 8),
            LoweredType::Tensor { .. } => {
                ffi::mlirTypeParseGet(context.raw, ffi::string_ref(&crate::emit::mlir_type(ty)?))
            }
            unsupported => return Err(MlirError::UnsupportedType(unsupported.clone())),
        }
    })
}

fn operation_name(operation: ffi::MlirOperation) -> String {
    let identifier = unsafe { ffi::mlirOperationGetName(operation) };
    string(unsafe { ffi::mlirIdentifierStr(identifier) }).to_owned()
}

fn collect_children(operation: ffi::MlirOperation, output: &mut Vec<ffi::MlirOperation>) {
    let mut region = unsafe { ffi::mlirOperationGetFirstRegion(operation) };
    while !region.is_null() {
        let mut block = unsafe { ffi::mlirRegionGetFirstBlock(region) };
        while !block.is_null() {
            let mut child = unsafe { ffi::mlirBlockGetFirstOperation(block) };
            while !child.is_null() {
                output.push(child);
                child = unsafe { ffi::mlirOperationGetNextInBlock(child) };
            }
            block = unsafe { ffi::mlirBlockGetNextInRegion(block) };
        }
        region = unsafe { ffi::mlirRegionGetNextInOperation(region) };
    }
}

fn string(value: ffi::MlirStringRef) -> &'static str {
    let bytes = unsafe { std::slice::from_raw_parts(value.data.cast::<u8>(), value.length) };
    unsafe { std::str::from_utf8_unchecked(bytes) }
}

unsafe extern "C" fn append_printed_text(value: ffi::MlirStringRef, user_data: *mut c_void) {
    let output = unsafe { &mut *user_data.cast::<String>() };
    output.push_str(string(value));
}
