pub(crate) fn mlir_kernels(relu: bool, add: bool, matmul: bool) -> String {
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
    source
}
