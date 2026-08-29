#![allow(unsafe_code)]
#![allow(dead_code)]

#[path = "../../../../compiler/transforms/mlir/src/ffi.rs"]
mod ffi;
#[path = "../../../../compiler/transforms/mlir/src/native.rs"]
mod native;

use native::{
    AffineExpression, AffineMap, Attribute, BlockRef, DetachedBlock, Location, ModuleBuilder,
    Operation, OperationState, Region, Type, Value,
};
use std::ffi::{c_char, c_void, CStr};

struct Session {
    builder: ModuleBuilder,
    types: Vec<Box<Type>>,
    values: Vec<Box<Value>>,
    attributes: Vec<Box<Attribute>>,
    locations: Vec<Box<Option<Location>>>,
    regions: Vec<Box<Option<Region>>>,
    blocks: Vec<Box<Option<DetachedBlock>>>,
    block_refs: Vec<Box<BlockRef>>,
    operations: Vec<Box<Option<Operation>>>,
    affine_expressions: Vec<Box<AffineExpression>>,
    affine_maps: Vec<Box<AffineMap>>,
    type_drafts: Vec<Box<TypeDraft>>,
    array_drafts: Vec<Box<ArrayDraft>>,
    affine_map_drafts: Vec<Box<AffineMapDraft>>,
    operation_drafts: Vec<Box<OperationDraft>>,
    error: Option<String>,
}

struct TypeDraft {
    element: Type,
    dimensions: Vec<i64>,
    kind: i32,
    inputs: Vec<Type>,
    results: Vec<Type>,
}

struct ArrayDraft {
    elements: Vec<Attribute>,
}

struct AffineMapDraft {
    dimensions: usize,
    symbols: usize,
    results: Vec<AffineExpression>,
}

struct OperationDraft {
    dialect: String,
    mnemonic: String,
    results: Vec<Type>,
    operands: Vec<Value>,
    regions: Vec<*mut c_void>,
    attributes: Vec<(String, Attribute)>,
    location: Option<Location>,
}

impl Session {
    fn new() -> Result<Self, String> {
        Ok(Self {
            builder: ModuleBuilder::new().map_err(|error| error.to_string())?,
            types: Vec::new(),
            values: Vec::new(),
            attributes: Vec::new(),
            locations: Vec::new(),
            regions: Vec::new(),
            blocks: Vec::new(),
            block_refs: Vec::new(),
            operations: Vec::new(),
            affine_expressions: Vec::new(),
            affine_maps: Vec::new(),
            type_drafts: Vec::new(),
            array_drafts: Vec::new(),
            affine_map_drafts: Vec::new(),
            operation_drafts: Vec::new(),
            error: None,
        })
    }

    fn fail(&mut self, error: impl ToString) {
        if self.error.is_none() {
            self.error = Some(error.to_string());
        }
    }

    fn keep_type(&mut self, value: Type) -> *mut c_void {
        self.types.push(Box::new(value));
        self.types.last_mut().unwrap().as_mut() as *mut Type as *mut c_void
    }

    fn keep_value(&mut self, value: Value) -> *mut c_void {
        self.values.push(Box::new(value));
        self.values.last_mut().unwrap().as_mut() as *mut Value as *mut c_void
    }

    fn keep_attribute(&mut self, value: Attribute) -> *mut c_void {
        self.attributes.push(Box::new(value));
        self.attributes.last_mut().unwrap().as_mut() as *mut Attribute as *mut c_void
    }

    fn keep_operation(&mut self, value: Operation) -> *mut c_void {
        self.operations.push(Box::new(Some(value)));
        self.operations.last_mut().unwrap().as_mut() as *mut Option<Operation> as *mut c_void
    }
}

unsafe fn session<'a>(value: *mut c_void) -> &'a mut Session {
    unsafe { &mut *value.cast::<Session>() }
}

unsafe fn copied<T: Copy>(value: *mut c_void) -> T {
    unsafe { *value.cast::<T>() }
}

unsafe fn text(value: *const c_char) -> String {
    if value.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(value) }.to_string_lossy().into_owned()
}

#[no_mangle]
pub extern "C" fn __sev_mlir_session_create_v1() -> *mut c_void {
    Session::new()
        .map(|session| Box::into_raw(Box::new(session)).cast())
        .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_session_destroy_v1(value: *mut c_void) {
    if !value.is_null() {
        drop(unsafe { Box::from_raw(value.cast::<Session>()) });
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_session_ok_v1(value: *mut c_void) -> bool {
    !value.is_null() && unsafe { session(value) }.error.is_none()
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_session_verify_v1(value: *mut c_void) -> bool {
    if value.is_null() {
        return false;
    }
    let session = unsafe { session(value) };
    session.error.is_none() && session.builder.verify()
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_integer_v1(
    value: *mut c_void,
    bits: u32,
    signedness: i32,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let ty = match signedness {
        -1 => session.builder.integer_type(bits),
        0 => session.builder.unsigned_integer_type(bits),
        _ => session.builder.signed_integer_type(bits),
    };
    session.keep_type(ty)
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_float_v1(
    value: *mut c_void,
    bits: u32,
    brain: bool,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let ty = match (brain, bits) {
        (true, 16) => session.builder.bf16_type(),
        (false, 16) => session.builder.f16_type(),
        (false, 32) => session.builder.f32_type(),
        (false, 64) => session.builder.f64_type(),
        _ => {
            session.fail(format!("unsupported native floating type: brain={brain}, bits={bits}"));
            return std::ptr::null_mut();
        }
    };
    session.keep_type(ty)
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_index_v1(value: *mut c_void) -> *mut c_void {
    let session = unsafe { session(value) };
    let ty = session.builder.index_type();
    session.keep_type(ty)
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_pointer_v1(
    value: *mut c_void,
    address_space: u32,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let ty = session.builder.pointer_type(address_space);
    session.keep_type(ty)
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_draft_v1(
    value: *mut c_void,
    kind: i32,
    element: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let element = if element.is_null() {
        session.builder.index_type()
    } else {
        unsafe { copied::<Type>(element) }
    };
    session.type_drafts.push(Box::new(TypeDraft {
        element,
        dimensions: Vec::new(),
        kind,
        inputs: Vec::new(),
        results: Vec::new(),
    }));
    session.type_drafts.last_mut().unwrap().as_mut() as *mut TypeDraft as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_draft_dimension_v1(
    draft: *mut c_void,
    dimension: i64,
) {
    unsafe { &mut *draft.cast::<TypeDraft>() }
        .dimensions
        .push(dimension);
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_draft_input_v1(
    draft: *mut c_void,
    ty: *mut c_void,
) {
    unsafe { &mut *draft.cast::<TypeDraft>() }
        .inputs
        .push(unsafe { copied(ty) });
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_draft_result_v1(
    draft: *mut c_void,
    ty: *mut c_void,
) {
    unsafe { &mut *draft.cast::<TypeDraft>() }
        .results
        .push(unsafe { copied(ty) });
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_type_draft_finish_v1(
    value: *mut c_void,
    draft: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let draft = unsafe { &*draft.cast::<TypeDraft>() };
    let result = match draft.kind {
        0 => session.builder.ranked_tensor_type(&draft.dimensions, draft.element),
        1 => session.builder.unranked_tensor_type(draft.element),
        2 => session.builder.memref_type(&draft.dimensions, draft.element),
        3 => session.builder.unranked_memref_type(draft.element),
        4 => session.builder.function_type(&draft.inputs, &draft.results),
        _ => Err(native::NativeBuilderError("unknown native type draft kind".into())),
    };
    match result {
        Ok(ty) => session.keep_type(ty),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_location_v1(
    value: *mut c_void,
    file: *const c_char,
    line: u32,
    column: u32,
) -> *mut c_void {
    let session = unsafe { session(value) };
    match Location::file_line_column(unsafe { text(file) }, line, column) {
        Ok(location) => {
            session.locations.push(Box::new(Some(location)));
            session.locations.last_mut().unwrap().as_mut() as *mut Option<Location> as *mut c_void
        }
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_location_unknown_v1(value: *mut c_void) -> *mut c_void {
    let session = unsafe { session(value) };
    session.locations.push(Box::new(None));
    session.locations.last_mut().unwrap().as_mut() as *mut Option<Location> as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_region_v1(value: *mut c_void) -> *mut c_void {
    let session = unsafe { session(value) };
    session.regions.push(Box::new(Some(session.builder.region())));
    session.regions.last_mut().unwrap().as_mut() as *mut Option<Region> as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_block_v1(value: *mut c_void) -> *mut c_void {
    let session = unsafe { session(value) };
    session.blocks.push(Box::new(Some(session.builder.block())));
    session.blocks.last_mut().unwrap().as_mut() as *mut Option<DetachedBlock> as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_block_argument_v1(
    value: *mut c_void,
    block: *mut c_void,
    ty: *mut c_void,
    location: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let block = unsafe { &*block.cast::<Option<DetachedBlock>>() };
    let Some(block) = block.as_ref() else {
        session.fail("cannot add an argument to a transferred block");
        return std::ptr::null_mut();
    };
    let location = unsafe { &*location.cast::<Option<Location>>() }.as_ref();
    match session
        .builder
        .add_argument(block, unsafe { copied(ty) }, location)
    {
        Ok(value) => session.keep_value(value),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_region_append_block_v1(
    value: *mut c_void,
    region: *mut c_void,
    block: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let Some(region_ref) = (unsafe { &*region.cast::<Option<Region>>() }).as_ref() else {
        session.fail("cannot append a block to a transferred region");
        return std::ptr::null_mut();
    };
    let Some(block_value) = (unsafe { &mut *block.cast::<Option<DetachedBlock>>() }).take() else {
        session.fail("native block has already been transferred");
        return std::ptr::null_mut();
    };
    match session.builder.append_block(region_ref, block_value) {
        Ok(block_ref) => {
            session.block_refs.push(Box::new(block_ref));
            session.block_refs.last_mut().unwrap().as_mut() as *mut BlockRef as *mut c_void
        }
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_attribute_string_v1(
    value: *mut c_void,
    text_value: *const c_char,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let attribute = session.builder.string_attribute(&unsafe { text(text_value) });
    session.keep_attribute(attribute)
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_attribute_symbol_v1(
    value: *mut c_void,
    text_value: *const c_char,
) -> *mut c_void {
    let session = unsafe { session(value) };
    match session.builder.symbol_attribute(&unsafe { text(text_value) }) {
        Ok(attribute) => session.keep_attribute(attribute),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_attribute_integer_v1(
    value: *mut c_void,
    ty: *mut c_void,
    integer: i64,
) -> *mut c_void {
    let session = unsafe { session(value) };
    match session.builder.integer_attribute(unsafe { copied(ty) }, integer) {
        Ok(attribute) => session.keep_attribute(attribute),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_attribute_float_v1(
    value: *mut c_void,
    ty: *mut c_void,
    floating: f64,
) -> *mut c_void {
    let session = unsafe { session(value) };
    match session.builder.float_attribute(unsafe { copied(ty) }, floating) {
        Ok(attribute) => session.keep_attribute(attribute),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_attribute_boolean_v1(
    value: *mut c_void,
    enabled: bool,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let attribute = session.builder.boolean_attribute(enabled);
    session.keep_attribute(attribute)
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_attribute_type_v1(
    value: *mut c_void,
    ty: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    match session.builder.type_attribute(unsafe { copied(ty) }) {
        Ok(attribute) => session.keep_attribute(attribute),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_array_draft_v1(value: *mut c_void) -> *mut c_void {
    let session = unsafe { session(value) };
    session
        .array_drafts
        .push(Box::new(ArrayDraft { elements: vec![] }));
    session.array_drafts.last_mut().unwrap().as_mut() as *mut ArrayDraft as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_array_draft_push_v1(
    draft: *mut c_void,
    attribute: *mut c_void,
) {
    unsafe { &mut *draft.cast::<ArrayDraft>() }
        .elements
        .push(unsafe { copied(attribute) });
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_array_draft_finish_v1(
    value: *mut c_void,
    draft: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let draft = unsafe { &*draft.cast::<ArrayDraft>() };
    match session.builder.array_attribute(&draft.elements) {
        Ok(attribute) => session.keep_attribute(attribute),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_affine_dimension_v1(
    value: *mut c_void,
    position: usize,
) -> *mut c_void {
    let session = unsafe { session(value) };
    session
        .affine_expressions
        .push(Box::new(AffineExpression::Dimension(position)));
    session.affine_expressions.last_mut().unwrap().as_mut() as *mut AffineExpression as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_affine_symbol_v1(
    value: *mut c_void,
    position: usize,
) -> *mut c_void {
    let session = unsafe { session(value) };
    session
        .affine_expressions
        .push(Box::new(AffineExpression::Symbol(position)));
    session.affine_expressions.last_mut().unwrap().as_mut() as *mut AffineExpression as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_affine_constant_v1(
    value: *mut c_void,
    constant: i64,
) -> *mut c_void {
    let session = unsafe { session(value) };
    session
        .affine_expressions
        .push(Box::new(AffineExpression::Constant(constant)));
    session.affine_expressions.last_mut().unwrap().as_mut() as *mut AffineExpression as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_affine_binary_v1(
    value: *mut c_void,
    kind: i32,
    left: *mut c_void,
    right: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let left = Box::new(unsafe { &*left.cast::<AffineExpression>() }.clone());
    let right = Box::new(unsafe { &*right.cast::<AffineExpression>() }.clone());
    let expression = match kind {
        0 => AffineExpression::Add(left, right),
        1 => AffineExpression::Multiply(left, right),
        2 => AffineExpression::Modulo(left, right),
        3 => AffineExpression::FloorDivide(left, right),
        _ => AffineExpression::CeilDivide(left, right),
    };
    session.affine_expressions.push(Box::new(expression));
    session.affine_expressions.last_mut().unwrap().as_mut() as *mut AffineExpression as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_affine_map_draft_v1(
    value: *mut c_void,
    dimensions: usize,
    symbols: usize,
) -> *mut c_void {
    let session = unsafe { session(value) };
    session.affine_map_drafts.push(Box::new(AffineMapDraft {
        dimensions,
        symbols,
        results: vec![],
    }));
    session.affine_map_drafts.last_mut().unwrap().as_mut() as *mut AffineMapDraft as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_affine_map_draft_push_v1(
    draft: *mut c_void,
    expression: *mut c_void,
) {
    unsafe { &mut *draft.cast::<AffineMapDraft>() }
        .results
        .push(unsafe { &*expression.cast::<AffineExpression>() }.clone());
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_affine_map_draft_finish_v1(
    value: *mut c_void,
    draft: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let draft = unsafe { &*draft.cast::<AffineMapDraft>() };
    match AffineMap::new(draft.dimensions, draft.symbols, draft.results.clone()) {
        Ok(map) => {
            session.affine_maps.push(Box::new(map));
            let attribute = session
                .builder
                .affine_map_attribute(session.affine_maps.last().unwrap());
            session.keep_attribute(attribute)
        }
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_operation_draft_v1(
    value: *mut c_void,
    dialect: *const c_char,
    mnemonic: *const c_char,
    location: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    session.operation_drafts.push(Box::new(OperationDraft {
        dialect: unsafe { text(dialect) },
        mnemonic: unsafe { text(mnemonic) },
        results: vec![],
        operands: vec![],
        regions: vec![],
        attributes: vec![],
        location: unsafe { &*location.cast::<Option<Location>>() }.clone(),
    }));
    session.operation_drafts.last_mut().unwrap().as_mut() as *mut OperationDraft as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_operation_draft_result_v1(
    draft: *mut c_void,
    ty: *mut c_void,
) {
    unsafe { &mut *draft.cast::<OperationDraft>() }
        .results
        .push(unsafe { copied(ty) });
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_operation_draft_operand_v1(
    draft: *mut c_void,
    value: *mut c_void,
) {
    unsafe { &mut *draft.cast::<OperationDraft>() }
        .operands
        .push(unsafe { copied(value) });
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_operation_draft_region_v1(
    draft: *mut c_void,
    region: *mut c_void,
) {
    unsafe { &mut *draft.cast::<OperationDraft>() }
        .regions
        .push(region);
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_operation_draft_attribute_v1(
    draft: *mut c_void,
    name: *const c_char,
    attribute: *mut c_void,
) {
    unsafe { &mut *draft.cast::<OperationDraft>() }
        .attributes
        .push((unsafe { text(name) }, unsafe { copied(attribute) }));
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_operation_draft_finish_v1(
    value: *mut c_void,
    draft: *mut c_void,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let draft = unsafe { &*draft.cast::<OperationDraft>() };
    let mut state = match OperationState::new(&draft.dialect, &draft.mnemonic) {
        Ok(state) => state,
        Err(error) => {
            session.fail(error);
            return std::ptr::null_mut();
        }
    };
    for result in &draft.results {
        state = state.result(*result);
    }
    for operand in &draft.operands {
        state = state.operand(*operand);
    }
    for (name, attribute) in &draft.attributes {
        state = state.attribute(name, *attribute);
    }
    if let Some(location) = &draft.location {
        state = state.location(location.clone());
    }
    for region_pointer in &draft.regions {
        let region_pointer = *region_pointer;
        let Some(region) = (unsafe { &mut *region_pointer.cast::<Option<Region>>() }).take() else {
            session.fail("native region has already been transferred");
            return std::ptr::null_mut();
        };
        state = state.region(region);
    }
    match session.builder.create_operation(state) {
        Ok(operation) => session.keep_operation(operation),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_operation_result_v1(
    value: *mut c_void,
    operation: *mut c_void,
    index: usize,
) -> *mut c_void {
    let session = unsafe { session(value) };
    let operation = unsafe { &*operation.cast::<Option<Operation>>() };
    let Some(operation) = operation.as_ref() else {
        session.fail("cannot read a transferred operation result");
        return std::ptr::null_mut();
    };
    match session.builder.result(operation, index) {
        Ok(result) => session.keep_value(result),
        Err(error) => {
            session.fail(error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_block_append_operation_v1(
    value: *mut c_void,
    block: *mut c_void,
    operation: *mut c_void,
) {
    let session = unsafe { session(value) };
    let block = unsafe { copied::<BlockRef>(block) };
    let Some(operation) = (unsafe { &mut *operation.cast::<Option<Operation>>() }).take() else {
        session.fail("native operation has already been transferred");
        return;
    };
    if let Err(error) = session.builder.append_operation(block, operation) {
        session.fail(error);
    }
}

#[no_mangle]
pub unsafe extern "C" fn __sev_mlir_module_append_operation_v1(
    value: *mut c_void,
    operation: *mut c_void,
) {
    let session = unsafe { session(value) };
    let Some(operation) = (unsafe { &mut *operation.cast::<Option<Operation>>() }).take() else {
        session.fail("native operation has already been transferred");
        return;
    };
    if let Err(error) = session.builder.append_to_module(operation) {
        session.fail(error);
    }
}
