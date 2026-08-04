pub(crate) fn mlir_kernels(
    relu: bool,
    add: bool,
    matmul: bool,
    transpose: bool,
    scale: bool,
    softmax_rows: bool,
    layer_norm: bool,
    relu_backward: bool,
    softmax_backward: bool,
    layer_norm_backward: bool,
) -> String {
    let mut source = String::new();
    if relu {
        source.push_str(
            r#"  func.func @__sev_linalg_relu(%input: memref<?xf64>, %output: memref<?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%input : memref<?xf64>) outs(%output : memref<?xf64>) {
    ^bb0(%value: f64, %unused: f64):
      %zero = arith.constant 0.0 : f64
      %positive = arith.cmpf ogt, %value, %zero : f64
      %result = arith.select %positive, %value, %zero : f64
      linalg.yield %result : f64
    }
    return
  }
"#,
        );
    }
    if add {
        source.push_str(
            r#"  func.func @__sev_linalg_add(%left: memref<?xf64>, %right: memref<?xf64>, %output: memref<?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%left, %right : memref<?xf64>, memref<?xf64>) outs(%output : memref<?xf64>) {
    ^bb0(%left_value: f64, %right_value: f64, %unused: f64):
      %result = arith.addf %left_value, %right_value : f64
      linalg.yield %result : f64
    }
    return
  }
"#,
        );
    }
    if matmul {
        source.push_str(
            r#"  func.func @__sev_linalg_matmul(%left: memref<?x?xf64>, %right: memref<?x?xf64>, %output: memref<?x?xf64>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%left, %right : memref<?x?xf64>, memref<?x?xf64>) outs(%output : memref<?x?xf64>)
    return
  }
"#,
        );
    }
    if transpose {
        source.push_str(
            r#"  func.func @__sev_linalg_transpose(%input: memref<?x?xf64>, %output: memref<?x?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<?x?xf64>) outs(%output : memref<?x?xf64>) {
    ^bb0(%value: f64, %unused: f64):
      linalg.yield %value : f64
    }
    return
  }
"#,
        );
    }
    if scale {
        source.push_str(
            r#"  func.func @__sev_linalg_scale(%input: memref<?xf64>, %scale: f64, %output: memref<?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%input : memref<?xf64>) outs(%output : memref<?xf64>) {
    ^bb0(%value: f64, %unused: f64):
      %result = arith.mulf %value, %scale : f64
      linalg.yield %result : f64
    }
    return
  }
"#,
        );
    }
    if softmax_rows {
        source.push_str(
            r#"  func.func @__sev_linalg_softmax_rows(%input: memref<?x?xf64>, %output: memref<?x?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<?x?xf64>) outs(%output : memref<?x?xf64>) {
    ^bb0(%value: f64, %unused: f64):
      %row = linalg.index 0 : index
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %width = memref.dim %input, %one : memref<?x?xf64>
      %negative = arith.constant -1.7976931348623157e+308 : f64
      %maximum = scf.for %column = %zero to %width step %one iter_args(%current = %negative) -> (f64) {
        %candidate = memref.load %input[%row, %column] : memref<?x?xf64>
        %larger = arith.cmpf ogt, %candidate, %current : f64
        %next = arith.select %larger, %candidate, %current : f64
        scf.yield %next : f64
      }
      %sum = scf.for %column = %zero to %width step %one iter_args(%current = %negative) -> (f64) {
        %candidate = memref.load %input[%row, %column] : memref<?x?xf64>
        %centered_raw = arith.subf %candidate, %maximum : f64
        %minimum = arith.constant -2.0e+01 : f64
        %below_minimum = arith.cmpf olt, %centered_raw, %minimum : f64
        %centered = arith.select %below_minimum, %minimum, %centered_raw : f64
        %one_float = arith.constant 1.0 : f64
        %steps = arith.constant 1.024e+03 : f64
        %fraction = arith.divf %centered, %steps : f64
        %base = arith.addf %one_float, %fraction : f64
        %exp1 = arith.mulf %base, %base : f64
        %exp2 = arith.mulf %exp1, %exp1 : f64
        %exp3 = arith.mulf %exp2, %exp2 : f64
        %exp4 = arith.mulf %exp3, %exp3 : f64
        %exp5 = arith.mulf %exp4, %exp4 : f64
        %exp6 = arith.mulf %exp5, %exp5 : f64
        %exp7 = arith.mulf %exp6, %exp6 : f64
        %exp8 = arith.mulf %exp7, %exp7 : f64
        %exp9 = arith.mulf %exp8, %exp8 : f64
        %exponential = arith.mulf %exp9, %exp9 : f64
        %is_first = arith.cmpi eq, %column, %zero : index
        %next_sum = arith.addf %current, %exponential : f64
        %next = arith.select %is_first, %exponential, %next_sum : f64
        scf.yield %next : f64
      }
      %centered_raw = arith.subf %value, %maximum : f64
      %minimum = arith.constant -2.0e+01 : f64
      %below_minimum = arith.cmpf olt, %centered_raw, %minimum : f64
      %centered = arith.select %below_minimum, %minimum, %centered_raw : f64
      %one_float = arith.constant 1.0 : f64
      %steps = arith.constant 1.024e+03 : f64
      %fraction = arith.divf %centered, %steps : f64
      %base = arith.addf %one_float, %fraction : f64
      %exp1 = arith.mulf %base, %base : f64
      %exp2 = arith.mulf %exp1, %exp1 : f64
      %exp3 = arith.mulf %exp2, %exp2 : f64
      %exp4 = arith.mulf %exp3, %exp3 : f64
      %exp5 = arith.mulf %exp4, %exp4 : f64
      %exp6 = arith.mulf %exp5, %exp5 : f64
      %exp7 = arith.mulf %exp6, %exp6 : f64
      %exp8 = arith.mulf %exp7, %exp7 : f64
      %exp9 = arith.mulf %exp8, %exp8 : f64
      %numerator = arith.mulf %exp9, %exp9 : f64
      %result = arith.divf %numerator, %sum : f64
      linalg.yield %result : f64
    }
    return
  }
"#,
        );
    }
    if layer_norm {
        source.push_str(
            r#"  func.func @__sev_linalg_layer_norm(%input: memref<?x?xf64>, %epsilon: f64, %output: memref<?x?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<?x?xf64>) outs(%output : memref<?x?xf64>) {
    ^bb0(%value: f64, %unused: f64):
      %row = linalg.index 0 : index
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %width = memref.dim %input, %one : memref<?x?xf64>
      %zero_float = arith.constant 0.0 : f64
      %total = scf.for %column = %zero to %width step %one iter_args(%current = %zero_float) -> (f64) {
        %candidate = memref.load %input[%row, %column] : memref<?x?xf64>
        %next = arith.addf %current, %candidate : f64
        scf.yield %next : f64
      }
      %width_integer = arith.index_cast %width : index to i64
      %width_float = arith.sitofp %width_integer : i64 to f64
      %mean = arith.divf %total, %width_float : f64
      %variance_total = scf.for %column = %zero to %width step %one iter_args(%current = %zero_float) -> (f64) {
        %candidate = memref.load %input[%row, %column] : memref<?x?xf64>
        %centered = arith.subf %candidate, %mean : f64
        %square = arith.mulf %centered, %centered : f64
        %next = arith.addf %current, %square : f64
        scf.yield %next : f64
      }
      %variance = arith.divf %variance_total, %width_float : f64
      %stabilized = arith.addf %variance, %epsilon : f64
      %initial_guess = arith.constant 1.0 : f64
      %two = arith.constant 2.0 : f64
      %iterations = arith.constant 10 : index
      %deviation = scf.for %iteration = %zero to %iterations step %one iter_args(%guess = %initial_guess) -> (f64) {
        %quotient = arith.divf %stabilized, %guess : f64
        %sum = arith.addf %guess, %quotient : f64
        %next = arith.divf %sum, %two : f64
        scf.yield %next : f64
      }
      %centered = arith.subf %value, %mean : f64
      %result = arith.divf %centered, %deviation : f64
      linalg.yield %result : f64
    }
    return
  }
"#,
        );
    }
    if relu_backward {
        source.push_str(
            r#"  func.func @__sev_linalg_relu_backward(%input: memref<?xf64>, %upstream: memref<?xf64>, %output: memref<?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%input, %upstream : memref<?xf64>, memref<?xf64>) outs(%output : memref<?xf64>) {
    ^bb0(%input_value: f64, %upstream_value: f64, %unused: f64):
      %zero = arith.constant 0.0 : f64
      %active = arith.cmpf ogt, %input_value, %zero : f64
      %result = arith.select %active, %upstream_value, %zero : f64
      linalg.yield %result : f64
    }
    return
  }
"#,
        );
    }
    if softmax_backward {
        source.push_str(
            r#"  func.func @__sev_linalg_softmax_backward(%softmax: memref<?x?xf64>, %upstream: memref<?x?xf64>, %output: memref<?x?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%softmax, %upstream : memref<?x?xf64>, memref<?x?xf64>) outs(%output : memref<?x?xf64>) {
    ^bb0(%softmax_value: f64, %upstream_value: f64, %unused: f64):
      %row = linalg.index 0 : index
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %width = memref.dim %softmax, %one : memref<?x?xf64>
      %zero_float = arith.constant 0.0 : f64
      %dot = scf.for %column = %zero to %width step %one iter_args(%current = %zero_float) -> (f64) {
        %probability = memref.load %softmax[%row, %column] : memref<?x?xf64>
        %gradient = memref.load %upstream[%row, %column] : memref<?x?xf64>
        %product = arith.mulf %probability, %gradient : f64
        %next = arith.addf %current, %product : f64
        scf.yield %next : f64
      }
      %centered = arith.subf %upstream_value, %dot : f64
      %result = arith.mulf %softmax_value, %centered : f64
      linalg.yield %result : f64
    }
    return
  }
"#,
        );
    }
    if layer_norm_backward {
        source.push_str(
            r#"  func.func @__sev_linalg_layer_norm_backward(%input: memref<?x?xf64>, %upstream: memref<?x?xf64>, %epsilon: f64, %output: memref<?x?xf64>) attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %upstream : memref<?x?xf64>, memref<?x?xf64>) outs(%output : memref<?x?xf64>) {
    ^bb0(%input_value: f64, %upstream_value: f64, %unused: f64):
      %row = linalg.index 0 : index
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %width = memref.dim %input, %one : memref<?x?xf64>
      %zero_float = arith.constant 0.0 : f64
      %total = scf.for %column = %zero to %width step %one iter_args(%current = %zero_float) -> (f64) {
        %candidate = memref.load %input[%row, %column] : memref<?x?xf64>
        %next = arith.addf %current, %candidate : f64
        scf.yield %next : f64
      }
      %width_integer = arith.index_cast %width : index to i64
      %width_float = arith.sitofp %width_integer : i64 to f64
      %mean = arith.divf %total, %width_float : f64
      %variance_total = scf.for %column = %zero to %width step %one iter_args(%current = %zero_float) -> (f64) {
        %candidate = memref.load %input[%row, %column] : memref<?x?xf64>
        %centered = arith.subf %candidate, %mean : f64
        %square = arith.mulf %centered, %centered : f64
        %next = arith.addf %current, %square : f64
        scf.yield %next : f64
      }
      %variance = arith.divf %variance_total, %width_float : f64
      %stabilized = arith.addf %variance, %epsilon : f64
      %initial_guess = arith.constant 1.0 : f64
      %two = arith.constant 2.0 : f64
      %iterations = arith.constant 10 : index
      %deviation = scf.for %iteration = %zero to %iterations step %one iter_args(%guess = %initial_guess) -> (f64) {
        %quotient = arith.divf %stabilized, %guess : f64
        %sum = arith.addf %guess, %quotient : f64
        %next = arith.divf %sum, %two : f64
        scf.yield %next : f64
      }
      %upstream_total = scf.for %column = %zero to %width step %one iter_args(%current = %zero_float) -> (f64) {
        %gradient = memref.load %upstream[%row, %column] : memref<?x?xf64>
        %next = arith.addf %current, %gradient : f64
        scf.yield %next : f64
      }
      %upstream_mean = arith.divf %upstream_total, %width_float : f64
      %mixed_total = scf.for %column = %zero to %width step %one iter_args(%current = %zero_float) -> (f64) {
        %candidate = memref.load %input[%row, %column] : memref<?x?xf64>
        %gradient = memref.load %upstream[%row, %column] : memref<?x?xf64>
        %centered = arith.subf %candidate, %mean : f64
        %mixed = arith.mulf %gradient, %centered : f64
        %next = arith.addf %current, %mixed : f64
        scf.yield %next : f64
      }
      %mixed_mean = arith.divf %mixed_total, %width_float : f64
      %centered = arith.subf %input_value, %mean : f64
      %scaled_mixed = arith.mulf %centered, %mixed_mean : f64
      %variance_term = arith.divf %scaled_mixed, %stabilized : f64
      %without_mean = arith.subf %upstream_value, %upstream_mean : f64
      %adjusted = arith.subf %without_mean, %variance_term : f64
      %result = arith.divf %adjusted, %deviation : f64
      linalg.yield %result : f64
    }
    return
  }
"#,
        );
    }
    source
}
