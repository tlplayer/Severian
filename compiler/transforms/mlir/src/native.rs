//! Direct, in-memory construction of native MLIR operations.
//!
//! This is the bootstrap materializer below Severian's source-level
//! `MlirProgram`. It deliberately exposes operation names as data and never
//! accepts a textual MLIR module. Text printing belongs only to diagnostics.

#![allow(unsafe_code)]

use crate::ffi;
use std::fmt;
use std::ptr;

unsafe extern "C" fn collect_diagnostic(
    part: ffi::MlirStringRef,
    user_data: *mut core::ffi::c_void,
) {
    let bytes = unsafe { std::slice::from_raw_parts(part.data.cast::<u8>(), part.length) };
    let output = unsafe { &mut *user_data.cast::<String>() };
    output.push_str(&String::from_utf8_lossy(bytes));
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBuilderError(pub String);

impl fmt::Display for NativeBuilderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for NativeBuilderError {}

#[derive(Clone, Copy)]
pub struct Type(ffi::MlirType);

#[derive(Clone, Copy)]
pub struct Value(ffi::MlirValue);

#[derive(Clone, Copy)]
pub struct Attribute(ffi::MlirAttribute);

pub struct Region(ffi::MlirRegion);

#[derive(Clone, Copy)]
pub struct Block(ffi::MlirBlock);

pub struct Operation(ffi::MlirOperation);

pub struct OperationState {
    name: String,
    result_types: Vec<Type>,
    operands: Vec<Value>,
    regions: Vec<Region>,
    attributes: Vec<(String, Attribute)>,
}

impl OperationState {
    pub fn new(name: impl Into<String>) -> Result<Self, NativeBuilderError> {
        let name = name.into();
        let Some((dialect, operation)) = name.split_once('.') else {
            return Err(NativeBuilderError(
                "an MLIR operation name must be `dialect.mnemonic`".into(),
            ));
        };
        if dialect.is_empty() || operation.is_empty() {
            return Err(NativeBuilderError(
                "an MLIR operation name must contain a dialect and mnemonic".into(),
            ));
        }
        Ok(Self {
            name,
            result_types: Vec::new(),
            operands: Vec::new(),
            regions: Vec::new(),
            attributes: Vec::new(),
        })
    }

    pub fn result(mut self, ty: Type) -> Self {
        self.result_types.push(ty);
        self
    }

    pub fn operand(mut self, value: Value) -> Self {
        self.operands.push(value);
        self
    }

    pub fn region(mut self, region: Region) -> Self {
        self.regions.push(region);
        self
    }

    pub fn attribute(mut self, name: impl Into<String>, attribute: Attribute) -> Self {
        self.attributes.push((name.into(), attribute));
        self
    }
}

/// Owns the MLIR context and ModuleOp. All handles created by this builder are
/// valid only for its lifetime.
pub struct ModuleBuilder {
    context: ffi::MlirContext,
    module: ffi::MlirModule,
    location: ffi::MlirLocation,
}

impl ModuleBuilder {
    pub fn new() -> Result<Self, NativeBuilderError> {
        unsafe {
            let context = ffi::mlirContextCreate();
            if context.is_null() {
                return Err(NativeBuilderError("MLIR context creation failed".into()));
            }
            let registry = ffi::mlirDialectRegistryCreate();
            ffi::mlirRegisterAllDialects(registry);
            ffi::mlirContextAppendDialectRegistry(context, registry);
            ffi::mlirDialectRegistryDestroy(registry);
            ffi::mlirContextLoadAllAvailableDialects(context);
            ffi::mlirRegisterAllPasses();
            let location = ffi::mlirLocationUnknownGet(context);
            let module = ffi::mlirModuleCreateEmpty(location);
            if module.is_null() {
                ffi::mlirContextDestroy(context);
                return Err(NativeBuilderError("MLIR ModuleOp creation failed".into()));
            }
            Ok(Self {
                context,
                module,
                location,
            })
        }
    }

    pub fn integer_type(&self, bits: u32) -> Type {
        Type(unsafe { ffi::mlirIntegerTypeGet(self.context, bits) })
    }

    pub fn index_type(&self) -> Type {
        Type(unsafe { ffi::mlirIndexTypeGet(self.context) })
    }

    pub fn bf16_type(&self) -> Type {
        Type(unsafe { ffi::mlirBF16TypeGet(self.context) })
    }

    pub fn f32_type(&self) -> Type {
        Type(unsafe { ffi::mlirF32TypeGet(self.context) })
    }

    pub fn ranked_tensor_type(&self, dimensions: &[i64], element: Type) -> Type {
        Type(unsafe {
            ffi::mlirRankedTensorTypeGet(
                dimensions.len() as isize,
                dimensions.as_ptr(),
                element.0,
                ffi::MlirAttribute { ptr: ptr::null() },
            )
        })
    }

    pub fn function_type(&self, inputs: &[Type], results: &[Type]) -> Type {
        let inputs = inputs.iter().map(|ty| ty.0).collect::<Vec<_>>();
        let results = results.iter().map(|ty| ty.0).collect::<Vec<_>>();
        Type(unsafe {
            ffi::mlirFunctionTypeGet(
                self.context,
                inputs.len() as isize,
                inputs.as_ptr(),
                results.len() as isize,
                results.as_ptr(),
            )
        })
    }

    pub fn string_attribute(&self, value: &str) -> Attribute {
        Attribute(unsafe { ffi::mlirStringAttrGet(self.context, ffi::string_ref(value)) })
    }

    pub fn integer_attribute(&self, ty: Type, value: i64) -> Attribute {
        Attribute(unsafe { ffi::mlirIntegerAttrGet(ty.0, value) })
    }

    pub fn type_attribute(&self, ty: Type) -> Attribute {
        Attribute(unsafe { ffi::mlirTypeAttrGet(ty.0) })
    }

    pub fn region(&self) -> Region {
        Region(unsafe { ffi::mlirRegionCreate() })
    }

    pub fn block(&self) -> Block {
        Block(unsafe { ffi::mlirBlockCreate(0, ptr::null(), ptr::null()) })
    }

    pub fn add_argument(&self, block: Block, ty: Type) -> Value {
        Value(unsafe { ffi::mlirBlockAddArgument(block.0, ty.0, self.location) })
    }

    pub fn append_block(&self, region: &Region, block: Block) {
        unsafe { ffi::mlirRegionAppendOwnedBlock(region.0, block.0) }
    }

    pub fn create_operation(
        &self,
        operation: OperationState,
    ) -> Result<Operation, NativeBuilderError> {
        let result_types = operation
            .result_types
            .iter()
            .map(|ty| ty.0)
            .collect::<Vec<_>>();
        let operands = operation
            .operands
            .iter()
            .map(|value| value.0)
            .collect::<Vec<_>>();
        let regions = operation
            .regions
            .into_iter()
            .map(|region| region.0)
            .collect::<Vec<_>>();
        let attributes = operation
            .attributes
            .iter()
            .map(|(name, attribute)| unsafe {
                ffi::mlirNamedAttributeGet(
                    ffi::mlirIdentifierGet(self.context, ffi::string_ref(name)),
                    attribute.0,
                )
            })
            .collect::<Vec<_>>();
        let mut state =
            unsafe { ffi::mlirOperationStateGet(ffi::string_ref(&operation.name), self.location) };
        unsafe {
            ffi::mlirOperationStateAddResults(
                &mut state,
                result_types.len() as isize,
                result_types.as_ptr(),
            );
            ffi::mlirOperationStateAddOperands(
                &mut state,
                operands.len() as isize,
                operands.as_ptr(),
            );
            ffi::mlirOperationStateAddOwnedRegions(
                &mut state,
                regions.len() as isize,
                regions.as_ptr(),
            );
            ffi::mlirOperationStateAddAttributes(
                &mut state,
                attributes.len() as isize,
                attributes.as_ptr(),
            );
            let created = ffi::mlirOperationCreate(&mut state);
            if created.is_null() {
                return Err(NativeBuilderError(format!(
                    "MLIR rejected construction of `{}`",
                    operation.name
                )));
            }
            Ok(Operation(created))
        }
    }

    pub fn result(&self, operation: &Operation, index: usize) -> Value {
        Value(unsafe { ffi::mlirOperationGetResult(operation.0, index as isize) })
    }

    pub fn append_operation(&self, block: Block, operation: Operation) {
        unsafe { ffi::mlirBlockAppendOwnedOperation(block.0, operation.0) }
    }

    pub fn append_to_module(&self, operation: Operation) {
        let body = unsafe { ffi::mlirModuleGetBody(self.module) };
        unsafe { ffi::mlirBlockAppendOwnedOperation(body, operation.0) }
    }

    pub fn verify(&self) -> bool {
        unsafe { ffi::mlirOperationVerify(ffi::mlirModuleGetOperation(self.module)) }
    }

    /// Runs a pass pipeline directly on the live ModuleOp. The pipeline string
    /// selects registered pass objects; it is not MLIR assembly and never
    /// serializes the module.
    pub fn run_pass_pipeline(&mut self, pipeline: &str) -> Result<(), NativeBuilderError> {
        let manager = unsafe { ffi::mlirPassManagerCreate(self.context) };
        if manager.is_null() {
            return Err(NativeBuilderError(
                "MLIR pass-manager creation failed".into(),
            ));
        }
        let mut diagnostic = String::new();
        let parsed = unsafe {
            ffi::mlirParsePassPipeline(
                ffi::mlirPassManagerGetAsOpPassManager(manager),
                ffi::string_ref(pipeline),
                collect_diagnostic,
                (&mut diagnostic as *mut String).cast(),
            )
        };
        if parsed.value == 0 {
            unsafe { ffi::mlirPassManagerDestroy(manager) };
            return Err(NativeBuilderError(format!(
                "invalid MLIR pass pipeline: {diagnostic}"
            )));
        }
        unsafe { ffi::mlirPassManagerEnableVerifier(manager, true) };
        let outcome = unsafe {
            ffi::mlirPassManagerRunOnOp(manager, ffi::mlirModuleGetOperation(self.module))
        };
        unsafe { ffi::mlirPassManagerDestroy(manager) };
        if outcome.value == 0 {
            return Err(NativeBuilderError(
                "MLIR pass pipeline rejected the in-memory module".into(),
            ));
        }
        Ok(())
    }
}

impl Drop for ModuleBuilder {
    fn drop(&mut self) {
        unsafe {
            ffi::mlirModuleDestroy(self.module);
            ffi::mlirContextDestroy(self.context);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_and_verifies_module_op_without_text_parsing() {
        let builder = ModuleBuilder::new().unwrap();
        let i32_type = builder.integer_type(32);
        let body = builder.region();
        let entry = builder.block();
        builder.append_block(&body, entry);

        let zero = builder
            .create_operation(
                OperationState::new("arith.constant")
                    .unwrap()
                    .result(i32_type)
                    .attribute("value", builder.integer_attribute(i32_type, 0)),
            )
            .unwrap();
        let zero_value = builder.result(&zero, 0);
        builder.append_operation(entry, zero);
        let returned = builder
            .create_operation(
                OperationState::new("func.return")
                    .unwrap()
                    .operand(zero_value),
            )
            .unwrap();
        builder.append_operation(entry, returned);

        let signature = builder.function_type(&[], &[i32_type]);
        let function = builder
            .create_operation(
                OperationState::new("func.func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("main"))
                    .attribute("function_type", builder.type_attribute(signature))
                    .region(body),
            )
            .unwrap();
        builder.append_to_module(function);

        assert!(builder.verify());
    }

    #[test]
    fn native_types_preserve_rank_and_dynamic_dimensions_as_data() {
        let builder = ModuleBuilder::new().unwrap();
        let tensor = builder.ranked_tensor_type(&[1, -1, 128], builder.bf16_type());
        let body = builder.region();
        let entry = builder.block();
        let argument = builder.add_argument(entry, tensor);
        builder.append_block(&body, entry);
        let returned = builder
            .create_operation(
                OperationState::new("func.return")
                    .unwrap()
                    .operand(argument),
            )
            .unwrap();
        builder.append_operation(entry, returned);
        let signature = builder.function_type(&[tensor], &[tensor]);
        let function = builder
            .create_operation(
                OperationState::new("func.func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("identity"))
                    .attribute("function_type", builder.type_attribute(signature))
                    .region(body),
            )
            .unwrap();
        builder.append_to_module(function);
        assert!(builder.verify());
    }

    #[test]
    fn pass_manager_transforms_the_live_module() {
        let mut builder = ModuleBuilder::new().unwrap();
        let body = builder.region();
        let entry = builder.block();
        builder.append_block(&body, entry);
        let returned = builder
            .create_operation(OperationState::new("func.return").unwrap())
            .unwrap();
        builder.append_operation(entry, returned);
        let signature = builder.function_type(&[], &[]);
        let function = builder
            .create_operation(
                OperationState::new("func.func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("main"))
                    .attribute("function_type", builder.type_attribute(signature))
                    .region(body),
            )
            .unwrap();
        builder.append_to_module(function);
        builder
            .run_pass_pipeline("builtin.module(canonicalize,cse)")
            .unwrap();
        assert!(builder.verify());
    }
}
