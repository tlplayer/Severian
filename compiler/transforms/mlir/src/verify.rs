use crate::emit::{artifact_symbol, MlirArtifact, MlirError};
use crate::ffi;
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
    let module = Module::parse(&context, &artifact.module, "artifact")?;
    let entry = module.sole_entry()?;
    if !module.operation_has_body(entry) {
        return Err(MlirError::EntryFunctionIsDeclaration);
    }
    module.verify("artifact module")?;
    module.verify_allowed_dialects(target)?;
    verify_entry_signature(&context, entry, &artifact.inputs, &artifact.outputs)?;
    Ok(VerifiedMlirArtifact {
        id,
        module: module.print(),
        target: target.triple.clone(),
    })
}

pub fn compose(
    normal: &str,
    artifacts: &[VerifiedMlirArtifact],
    target: &TargetSpec,
) -> Result<String, MlirError> {
    let context = Context::new();
    let module = Module::parse(&context, normal, "ordinary module")?;
    module.verify("ordinary module")?;
    module.verify_allowed_dialects(target)?;

    let symbol_table = SymbolTable::new(&module)?;
    let mut artifact_ids = BTreeSet::new();
    let mut composed_declarations = BTreeSet::new();
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

    module.verify("composed module")?;
    module.verify_allowed_dialects(target)?;
    Ok(module.print())
}

struct Context {
    raw: ffi::MlirContext,
}

impl Context {
    fn new() -> Self {
        unsafe {
            let registry = ffi::mlirDialectRegistryCreate();
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
                    count += 1;
                    entry = Some(current);
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
            "builtin", "arith", "async", "cf", "func", "linalg", "llvm", "math", "scf", "tensor",
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
    let attribute =
        unsafe { ffi::mlirOperationGetAttributeByName(operation, ffi::string_ref("sym_name")) };
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
            } => ffi::mlirTypeParseGet(context.raw, ffi::string_ref("f8E4M3FN")),
            LoweredType::Float {
                format: LoweredFloatFormat::Float8E5M2,
            } => ffi::mlirTypeParseGet(context.raw, ffi::string_ref("f8E5M2")),
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
            LoweredType::Tensor { .. } => ffi::mlirTypeParseGet(
                context.raw,
                ffi::string_ref(&crate::emit::mlir_type(ty)?),
            ),
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
