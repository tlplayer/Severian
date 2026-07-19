#![forbid(unsafe_code)]

use severian_hir::{Instruction, Program};
use severian_mlir::Module;
use std::fmt::Write;

pub fn lower(program: &Program) -> Module {
    let mut output = String::from("module {\n");

    let strings: Vec<_> = program
        .functions
        .iter()
        .flat_map(|function| function.instructions.iter())
        .map(|instruction| match instruction {
            Instruction::Print(value) => value,
        })
        .collect();

    for (index, value) in strings.iter().enumerate() {
        writeln!(
            output,
            "  llvm.mlir.global internal constant @__sev_str_{index}(\"{}\\00\")",
            escape_string(value)
        )
        .unwrap();
    }

    if !strings.is_empty() {
        output.push_str("  llvm.func @puts(!llvm.ptr) -> i32\n\n");
    }

    let mut string_index = 0;

    for function in &program.functions {
        writeln!(output, "  llvm.func @{}() -> i32 {{", function.name).unwrap();
        for instruction in &function.instructions {
            match instruction {
                Instruction::Print(_) => {
                    writeln!(
                        output,
                        "    %message_{string_index} = llvm.mlir.addressof @__sev_str_{string_index} : !llvm.ptr"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "    %status_{string_index} = llvm.call @puts(%message_{string_index}) : (!llvm.ptr) -> i32"
                    )
                    .unwrap();
                    string_index += 1;
                }
            }
        }
        output.push_str(
            "    %success = llvm.mlir.constant(0 : i32) : i32\n    llvm.return %success : i32\n  }\n",
        );
    }

    output.push_str("}\n");
    Module::new(output)
}

fn escape_string(value: &str) -> String {
    let mut escaped = String::new();
    for byte in value.as_bytes() {
        match byte {
            b' '..=b'!' | b'#'..=b'[' | b']'..=b'~' => escaped.push(*byte as char),
            _ => write!(escaped, "\\{byte:02X}").unwrap(),
        }
    }
    escaped
}
