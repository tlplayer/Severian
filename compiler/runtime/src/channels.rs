use crate::lowering_abi::LoweredValue;
use severian_hir::ValueType;
use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct ChannelSelectCase {
    pub channel: String,
    pub value_type: ValueType,
}

#[derive(Debug, Clone)]
pub struct ChannelSelectLowering {
    pub record: LoweredValue,
    pub selected_index: LoweredValue,
    pub boxed_value: LoweredValue,
    pub mlir: String,
}

pub fn emit_channel_create(
    result_name: &str,
    capacity: &str,
) -> (LoweredValue, String) {
    (
        LoweredValue::new(result_name, ValueType::Channel),
        format!(
            "    {result_name} = llvm.call @__sev_channel_create({capacity}) : (i64) -> !llvm.ptr\n"
        ),
    )
}

pub fn emit_channel_send(
    result_name: &str,
    boxed_value: &str,
    channel: &str,
) -> (LoweredValue, String) {
    (
        LoweredValue::new(result_name, ValueType::Any),
        format!(
            "    {result_name} = llvm.call @__sev_channel_send_ptr_async({boxed_value}, {channel}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n"
        ),
    )
}

pub fn emit_channel_receive(
    result_name: &str,
    channel: &str,
) -> (LoweredValue, String) {
    (
        LoweredValue::new(result_name, ValueType::Any),
        format!(
            "    {result_name} = llvm.call @__sev_channel_receive_ptr({channel}) : (!llvm.ptr) -> !llvm.ptr\n"
        ),
    )
}

/// Emits one blocking multi-channel select.
///
/// Runtime ABI:
/// `__sev_channel_select_ptr(channels, count)` returns a collection:
/// `[selected_index, boxed_value]`.
pub fn emit_channel_select(
    record_name: &str,
    array_name: &str,
    count_name: &str,
    index_name: &str,
    value_name: &str,
    cases: &[ChannelSelectCase],
) -> ChannelSelectLowering {
    let mut mlir = String::new();

    writeln!(
        mlir,
        "    {count_name} = llvm.mlir.constant({} : i64) : i64",
        cases.len()
    )
    .unwrap();

    writeln!(
        mlir,
        "    {array_name} = llvm.call @__sev_collection_new({count_name}) : (i64) -> !llvm.ptr"
    )
    .unwrap();

    for case in cases {
        writeln!(
            mlir,
            "    llvm.call @__sev_collection_push({array_name}, {}) : (!llvm.ptr, !llvm.ptr) -> ()",
            case.channel
        )
        .unwrap();
    }

    writeln!(
        mlir,
        "    {record_name} = llvm.call @__sev_channel_select_ptr({array_name}, {count_name}) : (!llvm.ptr, i64) -> !llvm.ptr"
    )
    .unwrap();

    let zero = format!("{record_name}_zero");
    let one = format!("{record_name}_one");
    let boxed_index = format!("{record_name}_boxed_index");

    writeln!(
        mlir,
        "    {zero} = llvm.mlir.constant(0 : i64) : i64"
    )
    .unwrap();
    writeln!(
        mlir,
        "    {one} = llvm.mlir.constant(1 : i64) : i64"
    )
    .unwrap();
    writeln!(
        mlir,
        "    {boxed_index} = llvm.call @__sev_collection_get({record_name}, {zero}) : (!llvm.ptr, i64) -> !llvm.ptr"
    )
    .unwrap();
    writeln!(
        mlir,
        "    {index_name} = llvm.call @__sev_unbox_i64({boxed_index}) : (!llvm.ptr) -> i64"
    )
    .unwrap();
    writeln!(
        mlir,
        "    {value_name} = llvm.call @__sev_collection_get({record_name}, {one}) : (!llvm.ptr, i64) -> !llvm.ptr"
    )
    .unwrap();

    ChannelSelectLowering {
        record: LoweredValue::new(record_name, ValueType::Any),
        selected_index: LoweredValue::new(index_name, ValueType::Int),
        boxed_value: LoweredValue::new(value_name, ValueType::Any),
        mlir,
    }
}
