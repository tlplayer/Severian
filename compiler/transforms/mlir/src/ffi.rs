#![allow(unsafe_code)]

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
handle!(MlirBlock, *mut c_void);
handle!(MlirContext, *mut c_void);
handle!(MlirDialectRegistry, *mut c_void);
handle!(MlirDialectHandle, *const c_void);
handle!(MlirIdentifier, *const c_void);
handle!(MlirModule, *const c_void);
handle!(MlirOperation, *mut c_void);
handle!(MlirRegion, *mut c_void);
handle!(MlirSymbolTable, *mut c_void);
handle!(MlirType, *const c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlirStringRef {
    pub data: *const c_char,
    pub length: usize,
}

pub type MlirStringCallback = unsafe extern "C" fn(MlirStringRef, *mut c_void);

#[cfg_attr(
    target_os = "linux",
    link(name = "severian_mlir_capi", kind = "static")
)]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPIIR", kind = "static"))]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPIArith", kind = "static"))]
#[cfg_attr(target_os = "macos", link(name = "MLIRCAPIRegisterEverything", kind = "static"))]
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

    pub fn mlirModuleCreateParse(context: MlirContext, source: MlirStringRef) -> MlirModule;
    pub fn mlirModuleDestroy(module: MlirModule);
    pub fn mlirModuleGetBody(module: MlirModule) -> MlirBlock;
    pub fn mlirModuleGetOperation(module: MlirModule) -> MlirOperation;

    pub fn mlirBlockGetFirstOperation(block: MlirBlock) -> MlirOperation;
    pub fn mlirBlockAppendOwnedOperation(block: MlirBlock, operation: MlirOperation);
    pub fn mlirOperationGetNextInBlock(operation: MlirOperation) -> MlirOperation;
    pub fn mlirOperationGetFirstRegion(operation: MlirOperation) -> MlirRegion;
    pub fn mlirRegionGetNextInOperation(region: MlirRegion) -> MlirRegion;
    pub fn mlirRegionGetFirstBlock(region: MlirRegion) -> MlirBlock;
    pub fn mlirBlockGetNextInRegion(block: MlirBlock) -> MlirBlock;
    pub fn mlirOperationGetName(operation: MlirOperation) -> MlirIdentifier;
    pub fn mlirOperationClone(operation: MlirOperation) -> MlirOperation;
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
    pub fn mlirOperationPrint(
        operation: MlirOperation,
        callback: MlirStringCallback,
        user_data: *mut c_void,
    );

    pub fn mlirIdentifierStr(identifier: MlirIdentifier) -> MlirStringRef;
    pub fn mlirStringAttrGet(context: MlirContext, value: MlirStringRef) -> MlirAttribute;
    pub fn mlirStringAttrGetValue(attribute: MlirAttribute) -> MlirStringRef;
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
    pub fn mlirBF16TypeGet(context: MlirContext) -> MlirType;
    pub fn mlirF16TypeGet(context: MlirContext) -> MlirType;
    pub fn mlirF32TypeGet(context: MlirContext) -> MlirType;
    pub fn mlirF64TypeGet(context: MlirContext) -> MlirType;

    pub fn mlirSymbolTableCreate(operation: MlirOperation) -> MlirSymbolTable;
    pub fn mlirSymbolTableDestroy(symbol_table: MlirSymbolTable);
    pub fn mlirSymbolTableLookup(
        symbol_table: MlirSymbolTable,
        name: MlirStringRef,
    ) -> MlirOperation;
    pub fn mlirSymbolTableErase(symbol_table: MlirSymbolTable, operation: MlirOperation);
}

pub fn string_ref(value: &str) -> MlirStringRef {
    MlirStringRef {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}
