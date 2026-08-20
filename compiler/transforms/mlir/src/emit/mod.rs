use severian_backend::{LoweredModule, Operation};

pub fn render(module: &LoweredModule) -> String {
    let mut output = String::from("module {\n  func.func @main() {\n");
    for operation in &module.operations {
        match operation {
            Operation::ConstantI64 { value, result } => output.push_str(&format!(
                "    %v{} = arith.constant {value} : i64\n",
                result.0
            )),
            Operation::AddI64 {
                left,
                right,
                result,
            } => output.push_str(&format!(
                "    %v{} = arith.addi %v{}, %v{} : i64\n",
                result.0, left.0, right.0
            )),
        }
    }
    output.push_str("    return\n  }\n}\n");
    output
}
