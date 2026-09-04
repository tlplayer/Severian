#![allow(unsafe_code)]

use core::cell::Cell;
use core::ffi::{c_char, c_void};

macro_rules! handle {
    ($name:ident, $pointer:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name {
            pub ptr: $pointer,
        }

        impl $name {
            #[allow(dead_code)]
            pub fn is_null(self) -> bool {
                self.ptr.is_null()
            }
        }
    };
}

handle!(MlirAttribute, *const c_void);
handle!(MlirAffineExpr, *const c_void);
handle!(MlirAffineMap, *const c_void);
handle!(MlirBlock, *mut c_void);
handle!(MlirContext, *mut c_void);
handle!(MlirDialectRegistry, *mut c_void);
handle!(MlirDialectHandle, *const c_void);
handle!(MlirDiagnostic, *mut c_void);
handle!(MlirIdentifier, *const c_void);
handle!(MlirLocation, *const c_void);
handle!(MlirModule, *const c_void);
handle!(MlirOperation, *mut c_void);
handle!(MlirOpPassManager, *mut c_void);
handle!(MlirPassManager, *mut c_void);
handle!(MlirRegion, *mut c_void);
handle!(MlirSymbolTable, *mut c_void);
handle!(MlirType, *const c_void);
handle!(MlirValue, *const c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlirNamedAttribute {
    pub name: MlirIdentifier,
    pub attribute: MlirAttribute,
}

#[repr(C)]
pub struct MlirOperationState {
    pub name: MlirStringRef,
    pub location: MlirLocation,
    pub n_results: isize,
    pub results: *mut MlirType,
    pub n_operands: isize,
    pub operands: *mut MlirValue,
    pub n_regions: isize,
    pub regions: *mut MlirRegion,
    pub n_successors: isize,
    pub successors: *mut MlirBlock,
    pub n_attributes: isize,
    pub attributes: *mut MlirNamedAttribute,
    pub enable_result_type_inference: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlirStringRef {
    pub data: *const c_char,
    pub length: usize,
}

pub type MlirStringCallback = unsafe extern "C" fn(MlirStringRef, *mut c_void);
pub type MlirDiagnosticHandler =
    unsafe extern "C" fn(MlirDiagnostic, *mut c_void) -> MlirLogicalResult;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlirLogicalResult {
    pub value: i8,
}

#[cfg_attr(target_os = "macos", link(name = "MLIRCAPIIR", kind = "static"))]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPIArith", kind = "static"))]
#[cfg_attr(
    target_os = "macos",
    link(name = "MLIRCAPIRegisterEverything", kind = "static")
)]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPIAsync", kind = "static"))]
#[cfg_attr(
    target_os = "macos",
    link(name = "MLIRCAPIControlFlow", kind = "static")
)]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPIFunc", kind = "static"))]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPILLVM", kind = "static"))]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPIMath", kind = "static"))]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPISCF", kind = "static"))]
#[cfg_attr(target_os = "macos", link(name = "MLIR", kind = "dylib"))]
unsafe extern "C" {
    pub fn mlirContextCreate() -> MlirContext;
    pub fn mlirContextDestroy(context: MlirContext);
    pub fn mlirContextSetAllowUnregisteredDialects(context: MlirContext, allow: bool);
    pub fn mlirContextAppendDialectRegistry(context: MlirContext, registry: MlirDialectRegistry);
    pub fn mlirContextLoadAllAvailableDialects(context: MlirContext);
    pub fn mlirDialectRegistryCreate() -> MlirDialectRegistry;
    pub fn mlirDialectRegistryDestroy(registry: MlirDialectRegistry);
    pub fn mlirGetDialectHandle__arith__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__async__() -> MlirDialectHandle;
    pub fn mlirRegisterAllDialects(registry: MlirDialectRegistry);
    pub fn mlirGetDialectHandle__cf__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__func__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__gpu__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__llvm__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__linalg__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__math__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__rocdl__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__scf__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__tensor__() -> MlirDialectHandle;
    pub fn mlirGetDialectHandle__vector__() -> MlirDialectHandle;
    pub fn mlirDialectHandleInsertDialect(handle: MlirDialectHandle, registry: MlirDialectRegistry);

    pub fn mlirLocationUnknownGet(context: MlirContext) -> MlirLocation;
    pub fn mlirLocationFileLineColGet(
        context: MlirContext,
        filename: MlirStringRef,
        line: u32,
        column: u32,
    ) -> MlirLocation;
    pub fn mlirModuleCreateEmpty(location: MlirLocation) -> MlirModule;
    #[link_name = "mlirModuleCreateParse"]
    fn mlir_module_create_parse_raw(context: MlirContext, source: MlirStringRef) -> MlirModule;
    pub fn mlirModuleDestroy(module: MlirModule);
    pub fn mlirModuleGetBody(module: MlirModule) -> MlirBlock;
    pub fn mlirModuleGetOperation(module: MlirModule) -> MlirOperation;

    pub fn mlirBlockGetFirstOperation(block: MlirBlock) -> MlirOperation;
    pub fn mlirBlockCreate(
        arguments: isize,
        types: *const MlirType,
        locations: *const MlirLocation,
    ) -> MlirBlock;
    pub fn mlirBlockAddArgument(
        block: MlirBlock,
        ty: MlirType,
        location: MlirLocation,
    ) -> MlirValue;
    pub fn mlirBlockAppendOwnedOperation(block: MlirBlock, operation: MlirOperation);
    pub fn mlirOperationGetNextInBlock(operation: MlirOperation) -> MlirOperation;
    pub fn mlirOperationGetFirstRegion(operation: MlirOperation) -> MlirRegion;
    pub fn mlirRegionGetNextInOperation(region: MlirRegion) -> MlirRegion;
    pub fn mlirRegionGetFirstBlock(region: MlirRegion) -> MlirBlock;
    pub fn mlirBlockGetNextInRegion(block: MlirBlock) -> MlirBlock;
    pub fn mlirOperationGetName(operation: MlirOperation) -> MlirIdentifier;
    pub fn mlirOperationClone(operation: MlirOperation) -> MlirOperation;
    pub fn mlirOperationStateGet(name: MlirStringRef, location: MlirLocation)
        -> MlirOperationState;
    pub fn mlirOperationStateAddResults(
        state: *mut MlirOperationState,
        count: isize,
        results: *const MlirType,
    );
    pub fn mlirOperationStateAddOperands(
        state: *mut MlirOperationState,
        count: isize,
        operands: *const MlirValue,
    );
    pub fn mlirOperationStateAddOwnedRegions(
        state: *mut MlirOperationState,
        count: isize,
        regions: *const MlirRegion,
    );
    pub fn mlirOperationStateAddSuccessors(
        state: *mut MlirOperationState,
        count: isize,
        successors: *const MlirBlock,
    );
    pub fn mlirOperationStateAddAttributes(
        state: *mut MlirOperationState,
        count: isize,
        attributes: *const MlirNamedAttribute,
    );
    pub fn mlirOperationStateEnableResultTypeInference(state: *mut MlirOperationState);
    pub fn mlirOperationCreate(state: *mut MlirOperationState) -> MlirOperation;
    pub fn mlirOperationDestroy(operation: MlirOperation);
    pub fn mlirOperationGetNumResults(operation: MlirOperation) -> isize;
    pub fn mlirOperationGetResult(operation: MlirOperation, position: isize) -> MlirValue;
    pub fn mlirOperationVerify(operation: MlirOperation) -> bool;
    pub fn mlirOperationGetAttributeByName(
        operation: MlirOperation,
        name: MlirStringRef,
    ) -> MlirAttribute;
    pub fn mlirOperationSetAttributeByName(
        operation: MlirOperation,
        name: MlirStringRef,
        attribute: MlirAttribute,
    );
    pub fn mlirOperationHasInherentAttributeByName(
        operation: MlirOperation,
        name: MlirStringRef,
    ) -> bool;
    pub fn mlirOperationSetInherentAttributeByName(
        operation: MlirOperation,
        name: MlirStringRef,
        attribute: MlirAttribute,
    );
    pub fn mlirOperationPrint(
        operation: MlirOperation,
        callback: MlirStringCallback,
        user_data: *mut c_void,
    );

    pub fn mlirIdentifierStr(identifier: MlirIdentifier) -> MlirStringRef;
    pub fn mlirIdentifierGet(context: MlirContext, value: MlirStringRef) -> MlirIdentifier;
    pub fn mlirNamedAttributeGet(
        name: MlirIdentifier,
        attribute: MlirAttribute,
    ) -> MlirNamedAttribute;
    pub fn mlirStringAttrGet(context: MlirContext, value: MlirStringRef) -> MlirAttribute;
    pub fn mlirAttributeGetNull() -> MlirAttribute;
    pub fn mlirIntegerAttrGet(ty: MlirType, value: i64) -> MlirAttribute;
    pub fn mlirFloatAttrDoubleGet(context: MlirContext, ty: MlirType, value: f64) -> MlirAttribute;
    pub fn mlirBoolAttrGet(context: MlirContext, value: i32) -> MlirAttribute;
    pub fn mlirArrayAttrGet(
        context: MlirContext,
        count: isize,
        elements: *const MlirAttribute,
    ) -> MlirAttribute;
    pub fn mlirFlatSymbolRefAttrGet(context: MlirContext, symbol: MlirStringRef) -> MlirAttribute;
    pub fn mlirAffineMapAttrGet(map: MlirAffineMap) -> MlirAttribute;
    pub fn mlirTypeAttrGet(ty: MlirType) -> MlirAttribute;
    pub fn mlirStringAttrGetValue(attribute: MlirAttribute) -> MlirStringRef;
    pub fn mlirIntegerAttrGetValueInt(attribute: MlirAttribute) -> i64;
    pub fn mlirAttributeIsAType(attribute: MlirAttribute) -> bool;
    pub fn mlirTypeAttrGetValue(attribute: MlirAttribute) -> MlirType;

    pub fn mlirTypeIsAFunction(ty: MlirType) -> bool;
    pub fn mlirFunctionTypeGetNumInputs(ty: MlirType) -> isize;
    pub fn mlirFunctionTypeGetNumResults(ty: MlirType) -> isize;
    pub fn mlirFunctionTypeGetInput(ty: MlirType, position: isize) -> MlirType;
    pub fn mlirFunctionTypeGetResult(ty: MlirType, position: isize) -> MlirType;
    pub fn mlirTypeEqual(left: MlirType, right: MlirType) -> bool;
    pub fn mlirTypeParseGet(context: MlirContext, source: MlirStringRef) -> MlirType;
    pub fn mlirIntegerTypeGet(context: MlirContext, bits: u32) -> MlirType;
    pub fn mlirIntegerTypeSignedGet(context: MlirContext, bits: u32) -> MlirType;
    pub fn mlirIntegerTypeUnsignedGet(context: MlirContext, bits: u32) -> MlirType;
    pub fn mlirIndexTypeGet(context: MlirContext) -> MlirType;
    pub fn mlirFunctionTypeGet(
        context: MlirContext,
        input_count: isize,
        inputs: *const MlirType,
        result_count: isize,
        results: *const MlirType,
    ) -> MlirType;
    pub fn mlirBF16TypeGet(context: MlirContext) -> MlirType;
    pub fn mlirF16TypeGet(context: MlirContext) -> MlirType;
    pub fn mlirF32TypeGet(context: MlirContext) -> MlirType;
    pub fn mlirF64TypeGet(context: MlirContext) -> MlirType;
    pub fn mlirLLVMPointerTypeGet(context: MlirContext, address_space: u32) -> MlirType;
    pub fn mlirRankedTensorTypeGet(
        rank: isize,
        shape: *const i64,
        element_type: MlirType,
        encoding: MlirAttribute,
    ) -> MlirType;
    pub fn mlirUnrankedTensorTypeGet(element_type: MlirType) -> MlirType;
    pub fn mlirMemRefTypeContiguousGet(
        element_type: MlirType,
        rank: isize,
        shape: *const i64,
        memory_space: MlirAttribute,
    ) -> MlirType;
    pub fn mlirUnrankedMemRefTypeGet(
        element_type: MlirType,
        memory_space: MlirAttribute,
    ) -> MlirType;

    pub fn mlirAffineDimExprGet(context: MlirContext, position: isize) -> MlirAffineExpr;
    pub fn mlirAffineSymbolExprGet(context: MlirContext, position: isize) -> MlirAffineExpr;
    pub fn mlirAffineConstantExprGet(context: MlirContext, value: i64) -> MlirAffineExpr;
    pub fn mlirAffineAddExprGet(left: MlirAffineExpr, right: MlirAffineExpr) -> MlirAffineExpr;
    pub fn mlirAffineMulExprGet(left: MlirAffineExpr, right: MlirAffineExpr) -> MlirAffineExpr;
    pub fn mlirAffineModExprGet(left: MlirAffineExpr, right: MlirAffineExpr) -> MlirAffineExpr;
    pub fn mlirAffineFloorDivExprGet(left: MlirAffineExpr, right: MlirAffineExpr)
        -> MlirAffineExpr;
    pub fn mlirAffineCeilDivExprGet(left: MlirAffineExpr, right: MlirAffineExpr) -> MlirAffineExpr;
    pub fn mlirAffineMapGet(
        context: MlirContext,
        dimensions: isize,
        symbols: isize,
        result_count: isize,
        results: *mut MlirAffineExpr,
    ) -> MlirAffineMap;

    pub fn mlirRegionCreate() -> MlirRegion;
    pub fn mlirRegionAppendOwnedBlock(region: MlirRegion, block: MlirBlock);

    pub fn mlirSymbolTableCreate(operation: MlirOperation) -> MlirSymbolTable;
    pub fn mlirSymbolTableDestroy(symbol_table: MlirSymbolTable);
    pub fn mlirSymbolTableLookup(
        symbol_table: MlirSymbolTable,
        name: MlirStringRef,
    ) -> MlirOperation;
    pub fn mlirSymbolTableErase(symbol_table: MlirSymbolTable, operation: MlirOperation);

    pub fn mlirRegisterAllPasses();
    pub fn mlirPassManagerCreate(context: MlirContext) -> MlirPassManager;
    pub fn mlirPassManagerDestroy(manager: MlirPassManager);
    pub fn mlirPassManagerEnableVerifier(manager: MlirPassManager, enable: bool);
    pub fn mlirPassManagerGetAsOpPassManager(manager: MlirPassManager) -> MlirOpPassManager;
    pub fn mlirParsePassPipeline(
        manager: MlirOpPassManager,
        pipeline: MlirStringRef,
        callback: MlirStringCallback,
        user_data: *mut c_void,
    ) -> MlirLogicalResult;
    pub fn mlirPassManagerRunOnOp(
        manager: MlirPassManager,
        operation: MlirOperation,
    ) -> MlirLogicalResult;

    pub fn mlirDiagnosticPrint(
        diagnostic: MlirDiagnostic,
        callback: MlirStringCallback,
        user_data: *mut c_void,
    );
    pub fn mlirContextAttachDiagnosticHandler(
        context: MlirContext,
        handler: MlirDiagnosticHandler,
        user_data: *mut c_void,
        delete_user_data: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> u64;
    pub fn mlirContextDetachDiagnosticHandler(context: MlirContext, id: u64);
}

thread_local! {
    static MODULE_PARSE_CALLS: Cell<usize> = const { Cell::new(0) };
}

pub unsafe fn module_create_parse(context: MlirContext, source: MlirStringRef) -> MlirModule {
    MODULE_PARSE_CALLS.with(|calls| calls.set(calls.get() + 1));
    unsafe { mlir_module_create_parse_raw(context, source) }
}

#[cfg(test)]
pub fn module_parse_calls() -> usize {
    MODULE_PARSE_CALLS.with(Cell::get)
}

pub fn string_ref(value: &str) -> MlirStringRef {
    MlirStringRef {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}
