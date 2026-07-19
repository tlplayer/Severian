use severian_hir::{Expression, Function, Instruction, Program, ValueType};

#[test]
fn lowers_hello_to_mlir_text() {
    let program = Program {
        functions: vec![Function {
            name: "main".into(),
            params: vec![],
            return_type: ValueType::Unit,
            instructions: vec![Instruction::Print(Expression::String(
                "hello, severian".into(),
            ))],
            tests: vec![],
        }],
    };

    assert_eq!(
        severian_lowering::lower(&program).as_str(),
        concat!(
            "module {\n",
            "  llvm.mlir.global internal constant @__sev_str_0(\"hello, severian\\00\")\n",
            "  llvm.func @puts(!llvm.ptr) -> i32\n\n",
            "  llvm.func @main() -> i32 {\n",
            "    %v0 = llvm.mlir.addressof @__sev_str_0 : !llvm.ptr\n",
            "    %v1 = llvm.call @puts(%v0) : (!llvm.ptr) -> i32\n",
            "    %v2 = llvm.mlir.constant(0 : i32) : i32\n",
            "    llvm.return %v2 : i32\n",
            "  }\n",
            "}\n",
        )
    );
}
