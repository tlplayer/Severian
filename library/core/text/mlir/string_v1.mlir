module attributes {severian.library_id = "core.text.string", severian.abi_version = 1 : i32} {
  func.func private @__sev_storage_new(i64, !llvm.ptr) -> !llvm.ptr
  func.func private @__sev_storage_release(!llvm.ptr)
  func.func private @strlen(!llvm.ptr) -> i64
  func.func private @strcmp(!llvm.ptr, !llvm.ptr) -> i32
  func.func private @memcpy(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  func.func private @abort()

  // Compatibility entry for the current pointer-valued source string. The
  // storage provider owns the allocation and accepts borrowed/literal pointers
  // at release boundaries without probing memory before those pointers.
  func.func @__sev_string_concat(%left: !llvm.ptr, %right: !llvm.ptr) -> !llvm.ptr {
    %left_length = func.call @strlen(%left) : (!llvm.ptr) -> i64
    %right_length = func.call @strlen(%right) : (!llvm.ptr) -> i64
    %payload = arith.addi %left_length, %right_length : i64
    %terminator = arith.constant 1 : i64
    %allocation_size = arith.addi %payload, %terminator : i64
    %overflow_payload = arith.cmpi ult, %payload, %left_length : i64
    %overflow_allocation = arith.cmpi ult, %allocation_size, %payload : i64
    %overflow = arith.ori %overflow_payload, %overflow_allocation : i1
    cf.cond_br %overflow, ^overflow, ^allocate

  ^overflow:
    func.call @abort() : () -> ()
    cf.br ^allocate

  ^allocate:
    %null = llvm.mlir.zero : !llvm.ptr
    %base = func.call @__sev_storage_new(%allocation_size, %null) : (i64, !llvm.ptr) -> !llvm.ptr
    %allocation_failed = llvm.icmp "eq" %base, %null : !llvm.ptr
    cf.cond_br %allocation_failed, ^allocation_failure, ^copy

  ^allocation_failure:
    func.call @abort() : () -> ()
    cf.br ^copy

  ^copy:
    %data = llvm.getelementptr %base[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %after_left = llvm.getelementptr %data[%left_length] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %right_with_terminator = arith.addi %right_length, %terminator : i64
    %left_copy = func.call @memcpy(%data, %left, %left_length) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %right_copy = func.call @memcpy(%after_left, %right, %right_with_terminator) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    return %data : !llvm.ptr
  }

  func.func @__sev_string_compare(%left: !llvm.ptr, %right: !llvm.ptr) -> i32 {
    %result = func.call @strcmp(%left, %right) : (!llvm.ptr, !llvm.ptr) -> i32
    return %result : i32
  }

  func.func @__sev_string_release(%value: !llvm.ptr) {
    %null = llvm.mlir.zero : !llvm.ptr
    %empty = llvm.icmp "eq" %value, %null : !llvm.ptr
    cf.cond_br %empty, ^done, ^release

  ^release:
    func.call @__sev_storage_release(%value) : (!llvm.ptr) -> ()
    cf.br ^done

  ^done:
    return
  }

  // Byte-storage boundaries for the source-owned UTF-8 algorithms. These use
  // the same managed, NUL-terminated ABI as the other v1 String exports.
  func.func @__sev_text_codepoint(%value: i32) -> i64 {
    %result = arith.extui %value : i32 to i64
    return %result : i64
  }

  func.func @__sev_text_character(%value: i64) -> i32 {
    %result = arith.trunci %value : i64 to i32
    return %result : i32
  }

  func.func @__sev_text_size(%value: !llvm.ptr) -> i64 {
    %size = func.call @strlen(%value) : (!llvm.ptr) -> i64
    return %size : i64
  }

  func.func @__sev_text_load(%value: !llvm.ptr, %offset: i64) -> i8 {
    %address = llvm.getelementptr %value[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %byte = llvm.load %address : !llvm.ptr -> i8
    return %byte : i8
  }

  func.func @__sev_text_store(%byte: i8, %value: !llvm.ptr, %offset: i64) {
    %address = llvm.getelementptr %value[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %byte, %address : i8, !llvm.ptr
    return
  }

  func.func @__sev_text_allocate(%size: i64) -> !llvm.ptr {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %bytes = arith.addi %size, %one : i64
    %valid = arith.cmpi sge, %size, %zero : i64
    %room = arith.cmpi sgt, %bytes, %size : i64
    %safe = arith.andi %valid, %room : i1
    cf.assert %safe, "invalid string allocation size"
    %null = llvm.mlir.zero : !llvm.ptr
    %data = func.call @__sev_storage_new(%bytes, %null) : (i64, !llvm.ptr) -> !llvm.ptr
    %allocated = llvm.icmp "ne" %data, %null : !llvm.ptr
    cf.assert %allocated, "string allocation failed"
    %end = llvm.getelementptr %data[%size] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %terminator = arith.constant 0 : i8
    llvm.store %terminator, %end : i8, !llvm.ptr
    cf.br ^fill(%zero : i64)
  ^fill(%position: i64):
    %more = arith.cmpi slt, %position, %size : i64
    cf.cond_br %more, ^byte, ^done
  ^byte:
    %address = llvm.getelementptr %data[%position] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %initial = arith.constant 1 : i8
    llvm.store %initial, %address : i8, !llvm.ptr
    %next = arith.addi %position, %one : i64
    cf.br ^fill(%next : i64)
  ^done:
    return %data : !llvm.ptr
  }
}
