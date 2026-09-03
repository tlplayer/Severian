//! Direct, in-memory construction of native MLIR operations.
//!
//! This is the bootstrap materializer below Severian's source-level
//! `MlirProgram`. It deliberately exposes operation names as data and never
//! accepts a textual MLIR module. Text printing belongs only to diagnostics.

#![allow(unsafe_code)]

use crate::ffi;
use std::fmt;
use std::ptr;
use std::sync::Mutex;

unsafe extern "C" fn collect_diagnostic(
    part: ffi::MlirStringRef,
    user_data: *mut core::ffi::c_void,
) {
    let bytes = unsafe { std::slice::from_raw_parts(part.data.cast::<u8>(), part.length) };
    let output = unsafe { &mut *user_data.cast::<String>() };
    output.push_str(&String::from_utf8_lossy(bytes));
}

unsafe extern "C" fn collect_context_diagnostic(
    diagnostic: ffi::MlirDiagnostic,
    user_data: *mut core::ffi::c_void,
) -> ffi::MlirLogicalResult {
    let mut message = String::new();
    unsafe {
        ffi::mlirDiagnosticPrint(
            diagnostic,
            collect_diagnostic,
            (&mut message as *mut String).cast(),
        )
    };
    let diagnostics = unsafe { &*user_data.cast::<Mutex<Vec<String>>>() };
    diagnostics
        .lock()
        .expect("native MLIR diagnostic lock was poisoned")
        .push(message);
    ffi::MlirLogicalResult { value: 1 }
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
pub struct Type(ffi::MlirType, ffi::MlirContext);

#[derive(Clone, Copy)]
pub struct Value(ffi::MlirValue, ffi::MlirContext);

#[derive(Clone, Copy)]
pub struct Attribute(ffi::MlirAttribute, ffi::MlirContext);

pub struct Region(ffi::MlirRegion, ffi::MlirContext);

pub struct DetachedBlock(ffi::MlirBlock, ffi::MlirContext);

#[derive(Clone, Copy)]
pub struct BlockRef(ffi::MlirBlock, ffi::MlirContext);

pub struct Operation {
    raw: ffi::MlirOperation,
    result_count: usize,
    context: ffi::MlirContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    file: String,
    line: u32,
    column: u32,
}

impl Location {
    pub fn file_line_column(
        file: impl Into<String>,
        line: u32,
        column: u32,
    ) -> Result<Self, NativeBuilderError> {
        let file = file.into();
        if file.is_empty() || line == 0 || column == 0 {
            return Err(NativeBuilderError(
                "an MLIR source location requires a file and one-based line/column".into(),
            ));
        }
        Ok(Self { file, line, column })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineExpression {
    Dimension(usize),
    Symbol(usize),
    Constant(i64),
    Add(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
    Modulo(Box<Self>, Box<Self>),
    FloorDivide(Box<Self>, Box<Self>),
    CeilDivide(Box<Self>, Box<Self>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineMap {
    dimensions: usize,
    symbols: usize,
    results: Vec<AffineExpression>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationName {
    dialect: String,
    mnemonic: String,
}

impl OperationName {
    pub fn new(
        dialect: impl Into<String>,
        mnemonic: impl Into<String>,
    ) -> Result<Self, NativeBuilderError> {
        fn valid_segment(segment: &str) -> bool {
            let mut characters = segment.chars();
            characters
                .next()
                .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
                && characters.all(|character| {
                    character.is_ascii_alphanumeric() || character == '_' || character == '$'
                })
        }
        let dialect = dialect.into();
        let mnemonic = mnemonic.into();
        if !valid_segment(&dialect) || !mnemonic.split('.').all(valid_segment) {
            return Err(NativeBuilderError(
                "an MLIR operation name requires separate dialect and mnemonic fields".into(),
            ));
        }
        Ok(Self { dialect, mnemonic })
    }
}

impl fmt::Display for OperationName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.dialect, self.mnemonic)
    }
}

impl AffineMap {
    pub fn new(
        dimensions: usize,
        symbols: usize,
        results: Vec<AffineExpression>,
    ) -> Result<Self, NativeBuilderError> {
        fn validate(expression: &AffineExpression, dimensions: usize, symbols: usize) -> bool {
            match expression {
                AffineExpression::Dimension(position) => *position < dimensions,
                AffineExpression::Symbol(position) => *position < symbols,
                AffineExpression::Constant(_) => true,
                AffineExpression::Add(left, right)
                | AffineExpression::Multiply(left, right)
                | AffineExpression::Modulo(left, right)
                | AffineExpression::FloorDivide(left, right)
                | AffineExpression::CeilDivide(left, right) => {
                    validate(left, dimensions, symbols) && validate(right, dimensions, symbols)
                }
            }
        }
        if results
            .iter()
            .any(|expression| !validate(expression, dimensions, symbols))
        {
            return Err(NativeBuilderError(
                "affine expression references an out-of-range dimension or symbol".into(),
            ));
        }
        Ok(Self {
            dimensions,
            symbols,
            results,
        })
    }
}

pub struct OperationState {
    name: OperationName,
    result_types: Vec<Type>,
    operands: Vec<Value>,
    regions: Vec<Region>,
    attributes: Vec<(String, Attribute)>,
    location: Option<Location>,
}

impl OperationState {
    pub fn new(
        dialect: impl Into<String>,
        mnemonic: impl Into<String>,
    ) -> Result<Self, NativeBuilderError> {
        let name = OperationName::new(dialect, mnemonic)?;
        Ok(Self {
            name,
            result_types: Vec::new(),
            operands: Vec::new(),
            regions: Vec::new(),
            attributes: Vec::new(),
            location: None,
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

    pub fn location(mut self, location: Location) -> Self {
        self.location = Some(location);
        self
    }
}

/// Owns the MLIR context and ModuleOp. All handles created by this builder are
/// valid only for its lifetime.
pub struct ModuleBuilder {
    context: ffi::MlirContext,
    module: ffi::MlirModule,
    location: ffi::MlirLocation,
    diagnostics: Box<Mutex<Vec<String>>>,
    diagnostic_handler: u64,
}

impl ModuleBuilder {
    fn owns_context(&self, context: ffi::MlirContext) -> bool {
        self.context.ptr == context.ptr
    }

    fn foreign_handle(&self, kind: &str) -> NativeBuilderError {
        NativeBuilderError(format!(
            "a native MLIR {kind} belongs to a different context"
        ))
    }

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
            let diagnostics = Box::new(Mutex::new(Vec::new()));
            let diagnostic_handler = ffi::mlirContextAttachDiagnosticHandler(
                context,
                collect_context_diagnostic,
                (&*diagnostics as *const Mutex<Vec<String>>)
                    .cast_mut()
                    .cast(),
                None,
            );
            let location = ffi::mlirLocationUnknownGet(context);
            let module = ffi::mlirModuleCreateEmpty(location);
            if module.is_null() {
                ffi::mlirContextDetachDiagnosticHandler(context, diagnostic_handler);
                ffi::mlirContextDestroy(context);
                return Err(NativeBuilderError("MLIR ModuleOp creation failed".into()));
            }
            Ok(Self {
                context,
                module,
                location,
                diagnostics,
                diagnostic_handler,
            })
        }
    }

    pub fn integer_type(&self, bits: u32) -> Type {
        Type(
            unsafe { ffi::mlirIntegerTypeGet(self.context, bits) },
            self.context,
        )
    }

    pub fn signed_integer_type(&self, bits: u32) -> Type {
        Type(
            unsafe { ffi::mlirIntegerTypeSignedGet(self.context, bits) },
            self.context,
        )
    }

    pub fn unsigned_integer_type(&self, bits: u32) -> Type {
        Type(
            unsafe { ffi::mlirIntegerTypeUnsignedGet(self.context, bits) },
            self.context,
        )
    }

    pub fn index_type(&self) -> Type {
        Type(unsafe { ffi::mlirIndexTypeGet(self.context) }, self.context)
    }

    pub fn bf16_type(&self) -> Type {
        Type(unsafe { ffi::mlirBF16TypeGet(self.context) }, self.context)
    }

    pub fn f16_type(&self) -> Type {
        Type(unsafe { ffi::mlirF16TypeGet(self.context) }, self.context)
    }

    pub fn f32_type(&self) -> Type {
        Type(unsafe { ffi::mlirF32TypeGet(self.context) }, self.context)
    }

    pub fn f64_type(&self) -> Type {
        Type(unsafe { ffi::mlirF64TypeGet(self.context) }, self.context)
    }

    pub fn pointer_type(&self, address_space: u32) -> Type {
        Type(
            unsafe { ffi::mlirLLVMPointerTypeGet(self.context, address_space) },
            self.context,
        )
    }

    pub fn ranked_tensor_type(
        &self,
        dimensions: &[i64],
        element: Type,
    ) -> Result<Type, NativeBuilderError> {
        if !self.owns_context(element.1) {
            return Err(self.foreign_handle("element type"));
        }
        if dimensions.iter().any(|dimension| *dimension < -1) {
            return Err(NativeBuilderError(
                "dynamic MLIR dimensions use exactly -1".into(),
            ));
        }
        Ok(Type(
            unsafe {
                ffi::mlirRankedTensorTypeGet(
                    dimensions.len() as isize,
                    dimensions.as_ptr(),
                    element.0,
                    ffi::mlirAttributeGetNull(),
                )
            },
            self.context,
        ))
    }

    pub fn unranked_tensor_type(&self, element: Type) -> Result<Type, NativeBuilderError> {
        if !self.owns_context(element.1) {
            return Err(self.foreign_handle("element type"));
        }
        Ok(Type(
            unsafe { ffi::mlirUnrankedTensorTypeGet(element.0) },
            self.context,
        ))
    }

    pub fn memref_type(
        &self,
        dimensions: &[i64],
        element: Type,
    ) -> Result<Type, NativeBuilderError> {
        if !self.owns_context(element.1) {
            return Err(self.foreign_handle("element type"));
        }
        if dimensions.iter().any(|dimension| *dimension < -1) {
            return Err(NativeBuilderError(
                "dynamic MLIR dimensions use exactly -1".into(),
            ));
        }
        Ok(Type(
            unsafe {
                ffi::mlirMemRefTypeContiguousGet(
                    element.0,
                    dimensions.len() as isize,
                    dimensions.as_ptr(),
                    ffi::mlirAttributeGetNull(),
                )
            },
            self.context,
        ))
    }

    pub fn unranked_memref_type(&self, element: Type) -> Result<Type, NativeBuilderError> {
        if !self.owns_context(element.1) {
            return Err(self.foreign_handle("element type"));
        }
        Ok(Type(
            unsafe { ffi::mlirUnrankedMemRefTypeGet(element.0, ffi::mlirAttributeGetNull()) },
            self.context,
        ))
    }

    pub fn function_type(
        &self,
        inputs: &[Type],
        results: &[Type],
    ) -> Result<Type, NativeBuilderError> {
        if inputs
            .iter()
            .chain(results)
            .any(|ty| !self.owns_context(ty.1))
        {
            return Err(self.foreign_handle("function type component"));
        }
        let inputs = inputs.iter().map(|ty| ty.0).collect::<Vec<_>>();
        let results = results.iter().map(|ty| ty.0).collect::<Vec<_>>();
        Ok(Type(
            unsafe {
                ffi::mlirFunctionTypeGet(
                    self.context,
                    inputs.len() as isize,
                    inputs.as_ptr(),
                    results.len() as isize,
                    results.as_ptr(),
                )
            },
            self.context,
        ))
    }

    pub fn string_attribute(&self, value: &str) -> Attribute {
        Attribute(
            unsafe { ffi::mlirStringAttrGet(self.context, ffi::string_ref(value)) },
            self.context,
        )
    }

    pub fn integer_attribute(&self, ty: Type, value: i64) -> Result<Attribute, NativeBuilderError> {
        if !self.owns_context(ty.1) {
            return Err(self.foreign_handle("integer attribute type"));
        }
        Ok(Attribute(
            unsafe { ffi::mlirIntegerAttrGet(ty.0, value) },
            self.context,
        ))
    }

    pub fn float_attribute(&self, ty: Type, value: f64) -> Result<Attribute, NativeBuilderError> {
        if !self.owns_context(ty.1) {
            return Err(self.foreign_handle("floating attribute type"));
        }
        Ok(Attribute(
            unsafe { ffi::mlirFloatAttrDoubleGet(self.context, ty.0, value) },
            self.context,
        ))
    }

    pub fn boolean_attribute(&self, value: bool) -> Attribute {
        Attribute(
            unsafe { ffi::mlirBoolAttrGet(self.context, i32::from(value)) },
            self.context,
        )
    }

    pub fn array_attribute(&self, elements: &[Attribute]) -> Result<Attribute, NativeBuilderError> {
        if elements
            .iter()
            .any(|attribute| !self.owns_context(attribute.1))
        {
            return Err(self.foreign_handle("array element attribute"));
        }
        let elements = elements
            .iter()
            .map(|attribute| attribute.0)
            .collect::<Vec<_>>();
        Ok(Attribute(
            unsafe {
                ffi::mlirArrayAttrGet(self.context, elements.len() as isize, elements.as_ptr())
            },
            self.context,
        ))
    }

    pub fn symbol_attribute(&self, symbol: &str) -> Result<Attribute, NativeBuilderError> {
        if symbol.is_empty() {
            return Err(NativeBuilderError(
                "an MLIR symbol reference cannot be empty".into(),
            ));
        }
        Ok(Attribute(
            unsafe { ffi::mlirFlatSymbolRefAttrGet(self.context, ffi::string_ref(symbol)) },
            self.context,
        ))
    }

    pub fn type_attribute(&self, ty: Type) -> Result<Attribute, NativeBuilderError> {
        if !self.owns_context(ty.1) {
            return Err(self.foreign_handle("type attribute value"));
        }
        Ok(Attribute(
            unsafe { ffi::mlirTypeAttrGet(ty.0) },
            self.context,
        ))
    }

    fn affine_expression(&self, expression: &AffineExpression) -> ffi::MlirAffineExpr {
        match expression {
            AffineExpression::Dimension(position) => unsafe {
                ffi::mlirAffineDimExprGet(self.context, *position as isize)
            },
            AffineExpression::Symbol(position) => unsafe {
                ffi::mlirAffineSymbolExprGet(self.context, *position as isize)
            },
            AffineExpression::Constant(value) => unsafe {
                ffi::mlirAffineConstantExprGet(self.context, *value)
            },
            AffineExpression::Add(left, right) => unsafe {
                ffi::mlirAffineAddExprGet(
                    self.affine_expression(left),
                    self.affine_expression(right),
                )
            },
            AffineExpression::Multiply(left, right) => unsafe {
                ffi::mlirAffineMulExprGet(
                    self.affine_expression(left),
                    self.affine_expression(right),
                )
            },
            AffineExpression::Modulo(left, right) => unsafe {
                ffi::mlirAffineModExprGet(
                    self.affine_expression(left),
                    self.affine_expression(right),
                )
            },
            AffineExpression::FloorDivide(left, right) => unsafe {
                ffi::mlirAffineFloorDivExprGet(
                    self.affine_expression(left),
                    self.affine_expression(right),
                )
            },
            AffineExpression::CeilDivide(left, right) => unsafe {
                ffi::mlirAffineCeilDivExprGet(
                    self.affine_expression(left),
                    self.affine_expression(right),
                )
            },
        }
    }

    pub fn affine_map_attribute(&self, map: &AffineMap) -> Attribute {
        let mut results = map
            .results
            .iter()
            .map(|expression| self.affine_expression(expression))
            .collect::<Vec<_>>();
        let map = unsafe {
            ffi::mlirAffineMapGet(
                self.context,
                map.dimensions as isize,
                map.symbols as isize,
                results.len() as isize,
                results.as_mut_ptr(),
            )
        };
        Attribute(unsafe { ffi::mlirAffineMapAttrGet(map) }, self.context)
    }

    fn native_location(&self, location: Option<&Location>) -> ffi::MlirLocation {
        match location {
            Some(location) => unsafe {
                ffi::mlirLocationFileLineColGet(
                    self.context,
                    ffi::string_ref(&location.file),
                    location.line,
                    location.column,
                )
            },
            None => self.location,
        }
    }

    pub fn region(&self) -> Region {
        Region(unsafe { ffi::mlirRegionCreate() }, self.context)
    }

    pub fn block(&self) -> DetachedBlock {
        DetachedBlock(
            unsafe { ffi::mlirBlockCreate(0, ptr::null(), ptr::null()) },
            self.context,
        )
    }

    pub fn add_argument(
        &self,
        block: &DetachedBlock,
        ty: Type,
        location: Option<&Location>,
    ) -> Result<Value, NativeBuilderError> {
        if !self.owns_context(block.1) || !self.owns_context(ty.1) {
            return Err(self.foreign_handle("block argument"));
        }
        Ok(Value(
            unsafe { ffi::mlirBlockAddArgument(block.0, ty.0, self.native_location(location)) },
            self.context,
        ))
    }

    pub fn append_block(
        &self,
        region: &Region,
        block: DetachedBlock,
    ) -> Result<BlockRef, NativeBuilderError> {
        if !self.owns_context(region.1) || !self.owns_context(block.1) {
            return Err(self.foreign_handle("region or block"));
        }
        let attached = BlockRef(block.0, self.context);
        unsafe { ffi::mlirRegionAppendOwnedBlock(region.0, block.0) }
        Ok(attached)
    }

    pub fn create_operation(
        &self,
        operation: OperationState,
    ) -> Result<Operation, NativeBuilderError> {
        let mut attribute_names = std::collections::BTreeSet::new();
        for (name, attribute) in &operation.attributes {
            if name.is_empty() {
                return Err(NativeBuilderError(
                    "an MLIR attribute name cannot be empty".into(),
                ));
            }
            if !attribute_names.insert(name) {
                return Err(NativeBuilderError(format!(
                    "MLIR operation `{}` contains duplicate attribute `{name}`",
                    operation.name
                )));
            }
            if !self.owns_context(attribute.1) {
                return Err(self.foreign_handle("attribute"));
            }
            if attribute.0.is_null() {
                return Err(NativeBuilderError(format!(
                    "MLIR operation `{}` contains a null attribute `{name}`",
                    operation.name
                )));
            }
        }
        if operation
            .result_types
            .iter()
            .any(|ty| !self.owns_context(ty.1))
        {
            return Err(self.foreign_handle("result type"));
        }
        if operation.result_types.iter().any(|ty| ty.0.is_null()) {
            return Err(NativeBuilderError(format!(
                "MLIR operation `{}` contains a null result type",
                operation.name
            )));
        }
        if operation
            .operands
            .iter()
            .any(|value| !self.owns_context(value.1))
        {
            return Err(self.foreign_handle("SSA operand"));
        }
        if operation.operands.iter().any(|value| value.0.is_null()) {
            return Err(NativeBuilderError(format!(
                "MLIR operation `{}` contains a missing SSA operand",
                operation.name
            )));
        }
        if operation
            .regions
            .iter()
            .any(|region| !self.owns_context(region.1))
        {
            return Err(self.foreign_handle("owned region"));
        }
        if operation.regions.iter().any(|region| region.0.is_null()) {
            return Err(NativeBuilderError(format!(
                "MLIR operation `{}` contains a null owned region",
                operation.name
            )));
        }
        let result_count = operation.result_types.len();
        let location = self.native_location(operation.location.as_ref());
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
        let operation_name = operation.name.to_string();
        let mut state =
            unsafe { ffi::mlirOperationStateGet(ffi::string_ref(&operation_name), location) };
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
            Ok(Operation {
                raw: created,
                result_count,
                context: self.context,
            })
        }
    }

    pub fn result(&self, operation: &Operation, index: usize) -> Result<Value, NativeBuilderError> {
        if index >= operation.result_count {
            return Err(NativeBuilderError(format!(
                "MLIR operation has {} results, but result {index} was requested",
                operation.result_count
            )));
        }
        if !self.owns_context(operation.context) {
            return Err(self.foreign_handle("operation"));
        }
        Ok(Value(
            unsafe { ffi::mlirOperationGetResult(operation.raw, index as isize) },
            self.context,
        ))
    }

    pub fn append_operation(
        &self,
        block: BlockRef,
        operation: Operation,
    ) -> Result<(), NativeBuilderError> {
        if !self.owns_context(block.1) || !self.owns_context(operation.context) {
            return Err(self.foreign_handle("block or operation"));
        }
        unsafe { ffi::mlirBlockAppendOwnedOperation(block.0, operation.raw) }
        Ok(())
    }

    pub fn append_to_module(&self, operation: Operation) -> Result<(), NativeBuilderError> {
        if !self.owns_context(operation.context) {
            return Err(self.foreign_handle("operation"));
        }
        let body = unsafe { ffi::mlirModuleGetBody(self.module) };
        unsafe { ffi::mlirBlockAppendOwnedOperation(body, operation.raw) }
        Ok(())
    }

    pub fn verify(&self) -> bool {
        unsafe { ffi::mlirOperationVerify(ffi::mlirModuleGetOperation(self.module)) }
    }

    pub fn take_diagnostics(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .diagnostics
                .lock()
                .expect("native MLIR diagnostic lock was poisoned"),
        )
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
            ffi::mlirContextDetachDiagnosticHandler(self.context, self.diagnostic_handler);
            ffi::mlirContextDestroy(self.context);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_and_verifies_module_op_without_text_parsing() {
        let parse_calls = crate::ffi::module_parse_calls();
        let builder = ModuleBuilder::new().unwrap();
        let i32_type = builder.integer_type(32);
        let body = builder.region();
        let entry = builder.block();
        let entry = builder.append_block(&body, entry).unwrap();

        let zero = builder
            .create_operation(
                OperationState::new("arith", "constant")
                    .unwrap()
                    .result(i32_type)
                    .attribute("value", builder.integer_attribute(i32_type, 0).unwrap()),
            )
            .unwrap();
        let zero_value = builder.result(&zero, 0).unwrap();
        builder.append_operation(entry, zero).unwrap();
        let returned = builder
            .create_operation(
                OperationState::new("func", "return")
                    .unwrap()
                    .operand(zero_value),
            )
            .unwrap();
        builder.append_operation(entry, returned).unwrap();

        let signature = builder.function_type(&[], &[i32_type]).unwrap();
        let function = builder
            .create_operation(
                OperationState::new("func", "func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("main"))
                    .attribute("function_type", builder.type_attribute(signature).unwrap())
                    .region(body),
            )
            .unwrap();
        builder.append_to_module(function).unwrap();

        assert!(builder.verify());
        assert_eq!(crate::ffi::module_parse_calls(), parse_calls);
    }

    #[test]
    fn native_types_preserve_rank_and_dynamic_dimensions_as_data() {
        let builder = ModuleBuilder::new().unwrap();
        let tensor = builder
            .ranked_tensor_type(&[1, -1, 128], builder.bf16_type())
            .unwrap();
        let body = builder.region();
        let entry = builder.block();
        let location = Location::file_line_column("tensor.sev", 4, 9).unwrap();
        let argument = builder
            .add_argument(&entry, tensor, Some(&location))
            .unwrap();
        let entry = builder.append_block(&body, entry).unwrap();
        let returned = builder
            .create_operation(
                OperationState::new("func", "return")
                    .unwrap()
                    .operand(argument),
            )
            .unwrap();
        builder.append_operation(entry, returned).unwrap();
        let signature = builder.function_type(&[tensor], &[tensor]).unwrap();
        let function = builder
            .create_operation(
                OperationState::new("func", "func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("identity"))
                    .attribute("function_type", builder.type_attribute(signature).unwrap())
                    .region(body),
            )
            .unwrap();
        builder.append_to_module(function).unwrap();
        assert!(builder.verify());
    }

    #[test]
    fn pass_manager_transforms_the_live_module() {
        let mut builder = ModuleBuilder::new().unwrap();
        let body = builder.region();
        let entry = builder.block();
        let entry = builder.append_block(&body, entry).unwrap();
        let returned = builder
            .create_operation(OperationState::new("func", "return").unwrap())
            .unwrap();
        builder.append_operation(entry, returned).unwrap();
        let signature = builder.function_type(&[], &[]).unwrap();
        let function = builder
            .create_operation(
                OperationState::new("func", "func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("main"))
                    .attribute("function_type", builder.type_attribute(signature).unwrap())
                    .region(body),
            )
            .unwrap();
        builder.append_to_module(function).unwrap();
        builder
            .run_pass_pipeline("builtin.module(canonicalize,cse)")
            .unwrap();
        assert!(builder.verify());
    }

    #[test]
    fn materializes_the_complete_type_and_attribute_contract() {
        let parse_calls = crate::ffi::module_parse_calls();
        let builder = ModuleBuilder::new().unwrap();
        let bf16_type = builder.bf16_type();
        let f16_type = builder.f16_type();
        let f32_type = builder.f32_type();
        let f64_type = builder.f64_type();
        let index = builder.index_type();
        let signless = builder.integer_type(32);
        let signed = builder.signed_integer_type(128);
        let unsigned = builder.unsigned_integer_type(8);
        let pointer = builder.pointer_type(3);
        let rank_zero_tensor = builder.ranked_tensor_type(&[], bf16_type).unwrap();
        let ranked_tensor = builder.ranked_tensor_type(&[2, -1], f16_type).unwrap();
        let unranked_tensor = builder.unranked_tensor_type(f16_type).unwrap();
        let ranked_memref = builder.memref_type(&[-1, 8], f64_type).unwrap();
        let unranked_memref = builder.unranked_memref_type(f64_type).unwrap();
        let signature = builder
            .function_type(
                &[
                    pointer,
                    index,
                    signless,
                    ranked_tensor,
                    rank_zero_tensor,
                    unranked_tensor,
                    ranked_memref,
                    unranked_memref,
                    bf16_type,
                    f32_type,
                    signed,
                    unsigned,
                ],
                &[],
            )
            .unwrap();
        let affine = AffineMap::new(
            2,
            1,
            vec![
                AffineExpression::Dimension(0),
                AffineExpression::Add(
                    Box::new(AffineExpression::Dimension(1)),
                    Box::new(AffineExpression::Symbol(0)),
                ),
                AffineExpression::Multiply(
                    Box::new(AffineExpression::Dimension(0)),
                    Box::new(AffineExpression::Constant(2)),
                ),
                AffineExpression::Modulo(
                    Box::new(AffineExpression::Dimension(1)),
                    Box::new(AffineExpression::Constant(4)),
                ),
                AffineExpression::FloorDivide(
                    Box::new(AffineExpression::Dimension(1)),
                    Box::new(AffineExpression::Constant(8)),
                ),
                AffineExpression::CeilDivide(
                    Box::new(AffineExpression::Dimension(1)),
                    Box::new(AffineExpression::Constant(16)),
                ),
            ],
        )
        .unwrap();
        let strings = builder
            .array_attribute(&[
                builder.string_attribute("parallel"),
                builder.string_attribute("reduction"),
            ])
            .unwrap();
        let declaration_body = builder.region();
        let declaration = builder
            .create_operation(
                OperationState::new("func", "func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("contract"))
                    .attribute("sym_visibility", builder.string_attribute("private"))
                    .attribute("function_type", builder.type_attribute(signature).unwrap())
                    .attribute(
                        "severian.integer",
                        builder
                            .integer_attribute(builder.integer_type(32), 7)
                            .unwrap(),
                    )
                    .attribute(
                        "severian.float",
                        builder.float_attribute(f64_type, 1.25).unwrap(),
                    )
                    .attribute("severian.boolean", builder.boolean_attribute(true))
                    .attribute("severian.array", strings)
                    .attribute(
                        "severian.symbol",
                        builder.symbol_attribute("contract").unwrap(),
                    )
                    .attribute("severian.affine_map", builder.affine_map_attribute(&affine))
                    .location(Location::file_line_column("contract.sev", 3, 5).unwrap())
                    .region(declaration_body),
            )
            .unwrap();
        builder.append_to_module(declaration).unwrap();
        assert!(builder.verify(), "{:?}", builder.take_diagnostics());
        assert_eq!(crate::ffi::module_parse_calls(), parse_calls);
    }

    #[test]
    fn invalid_native_state_fails_before_mlir_construction() {
        let builder = ModuleBuilder::new().unwrap();
        assert!(OperationState::new("func.return", "").is_err());
        assert!(OperationState::new("func", "bad name").is_err());
        assert!(OperationState::new("llvm", "mlir.global").is_ok());
        let duplicate = builder.create_operation(
            OperationState::new("func", "func")
                .unwrap()
                .attribute("sym_name", builder.string_attribute("first"))
                .attribute("sym_name", builder.string_attribute("second")),
        );
        assert!(duplicate.is_err());

        let constant = builder
            .create_operation(
                OperationState::new("arith", "constant")
                    .unwrap()
                    .result(builder.integer_type(32))
                    .attribute(
                        "value",
                        builder
                            .integer_attribute(builder.integer_type(32), 0)
                            .unwrap(),
                    ),
            )
            .unwrap();
        assert!(builder.result(&constant, 1).is_err());
        assert!(builder.take_diagnostics().is_empty());
    }

    #[test]
    fn native_verifier_diagnostics_are_provider_owned() {
        let builder = ModuleBuilder::new().unwrap();
        let signature = builder
            .function_type(&[], &[builder.integer_type(32)])
            .unwrap();
        let body = builder.region();
        let entry = builder.block();
        let entry = builder.append_block(&body, entry).unwrap();
        let returned = builder
            .create_operation(OperationState::new("func", "return").unwrap())
            .unwrap();
        builder.append_operation(entry, returned).unwrap();
        let function = builder
            .create_operation(
                OperationState::new("func", "func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("invalid"))
                    .attribute("function_type", builder.type_attribute(signature).unwrap())
                    .region(body),
            )
            .unwrap();
        builder.append_to_module(function).unwrap();
        assert!(!builder.verify());
        assert!(!builder.take_diagnostics().is_empty());
    }

    #[test]
    fn nested_regions_transfer_into_their_parent_operation_once() {
        let builder = ModuleBuilder::new().unwrap();
        let boolean = builder.integer_type(1);
        let function_body = builder.region();
        let function_entry = builder.block();
        let condition = builder
            .add_argument(&function_entry, boolean, None)
            .unwrap();
        let function_entry = builder
            .append_block(&function_body, function_entry)
            .unwrap();

        let then_region = builder.region();
        let then_block = builder.append_block(&then_region, builder.block()).unwrap();
        let then_yield = builder
            .create_operation(OperationState::new("scf", "yield").unwrap())
            .unwrap();
        builder.append_operation(then_block, then_yield).unwrap();

        let else_region = builder.region();
        let else_block = builder.append_block(&else_region, builder.block()).unwrap();
        let else_yield = builder
            .create_operation(OperationState::new("scf", "yield").unwrap())
            .unwrap();
        builder.append_operation(else_block, else_yield).unwrap();

        let conditional = builder
            .create_operation(
                OperationState::new("scf", "if")
                    .unwrap()
                    .operand(condition)
                    .region(then_region)
                    .region(else_region),
            )
            .unwrap();
        builder
            .append_operation(function_entry, conditional)
            .unwrap();
        let returned = builder
            .create_operation(OperationState::new("func", "return").unwrap())
            .unwrap();
        builder.append_operation(function_entry, returned).unwrap();

        let signature = builder.function_type(&[boolean], &[]).unwrap();
        let function = builder
            .create_operation(
                OperationState::new("func", "func")
                    .unwrap()
                    .attribute("sym_name", builder.string_attribute("branch"))
                    .attribute("function_type", builder.type_attribute(signature).unwrap())
                    .region(function_body),
            )
            .unwrap();
        builder.append_to_module(function).unwrap();
        assert!(builder.verify(), "{:?}", builder.take_diagnostics());
    }

    #[test]
    fn rejects_cross_context_handles_before_entering_mlir() {
        let first = ModuleBuilder::new().unwrap();
        let second = ModuleBuilder::new().unwrap();

        assert!(second
            .ranked_tensor_type(&[2, 2], first.f32_type())
            .is_err());

        let block = first.block();
        let value = first
            .add_argument(&block, first.integer_type(32), None)
            .unwrap();
        let operation = OperationState::new("func", "return")
            .unwrap()
            .operand(value);
        assert!(second.create_operation(operation).is_err());
        assert!(second.take_diagnostics().is_empty());
    }
}
