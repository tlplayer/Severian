use severian_hir::{Function, Instruction, Program};

#[test]
fn lowers_hello_to_mlir_text() {
    let program = Program {
        functions: vec![Function {
            name: "main".into(),
            instructions: vec![Instruction::Print("hello, severian".into())],
        }],
    };

    assert_eq!(
        severian_lowering::lower(&program).as_str(),
        concat!(
            "module {\n",
            "  llvm.mlir.global internal constant @__sev_str_0(\"hello, severian\\00\")\n",
            "  llvm.func @puts(!llvm.ptr) -> i32\n\n",
            "  llvm.func @main() -> i32 {\n",
            "    %message_0 = llvm.mlir.addressof @__sev_str_0 : !llvm.ptr\n",
            "    %status_0 = llvm.call @puts(%message_0) : (!llvm.ptr) -> i32\n",
            "    %success = llvm.mlir.constant(0 : i32) : i32\n",
            "    llvm.return %success : i32\n",
            "  }\n",
            "}\n",
        )
    );
}
