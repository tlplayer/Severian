use super::*;

impl LowerContext<'_> {
    pub(super) fn lower_binary_values(
        &mut self,
        (mut left, mut operand_type): (String, ValueType),
        op: BinaryOp,
        (mut right, mut right_type): (String, ValueType),
    ) -> (String, ValueType) {
        if op == BinaryOp::In && right_type == ValueType::Any {
            (right, right_type) = self.unbox_value((right, right_type), ValueType::List);
        }
        if op == BinaryOp::In && matches!(right_type, ValueType::List | ValueType::Set) {
            let left = self.box_value((left, operand_type));
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @__sev_set_contains({right}, {left}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            return (result, ValueType::Bool);
        }
        if op == BinaryOp::In && right_type == ValueType::Map {
            let left = self.box_value((left, operand_type));
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @__sev_map_contains({right}, {left}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            return (result, ValueType::Bool);
        }
        if op == BinaryOp::Add && operand_type == ValueType::List && right_type == ValueType::List {
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @__sev_collection_concat({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
            return (result, ValueType::List);
        }
        if matches!(op, BinaryOp::Equal | BinaryOp::NotEqual)
            && matches!(
                operand_type,
                ValueType::List | ValueType::Tuple | ValueType::Map | ValueType::Set
            )
            && operand_type == right_type
        {
            let equal = self.fresh_value();
            writeln!(self.output, "    {equal} = llvm.call @__sev_collection_equal({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            if op == BinaryOp::Equal {
                return (equal, ValueType::Bool);
            }
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1").unwrap();
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.xor {equal}, {one} : i1").unwrap();
            return (result, ValueType::Bool);
        }
        if op == BinaryOp::Add
            && operand_type == ValueType::String
            && right_type == ValueType::String
        {
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @__sev_string_concat({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
            return (result, ValueType::String);
        }
        if op == BinaryOp::Mul
            && matches!(
                (operand_type, right_type),
                (ValueType::List, ValueType::Int) | (ValueType::Int, ValueType::List)
            )
        {
            let (collection, count) = if operand_type == ValueType::List {
                (left, right)
            } else {
                (right, left)
            };
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @__sev_collection_repeat({collection}, {count}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
            return (result, ValueType::List);
        }
        if matches!(op, BinaryOp::Equal | BinaryOp::NotEqual)
            && operand_type == ValueType::String
            && right_type == ValueType::String
        {
            let equal = self.fresh_value();
            writeln!(self.output, "    {equal} = llvm.call @__sev_string_equal({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            if op == BinaryOp::Equal {
                return (equal, ValueType::Bool);
            }
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1").unwrap();
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.xor {equal}, {one} : i1").unwrap();
            return (result, ValueType::Bool);
        }
        if operand_type == ValueType::Any
            && right_type == ValueType::Any
            && matches!(
                op,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
            )
        {
            let function = match op {
                BinaryOp::Add => "__sev_value_add",
                BinaryOp::Sub => "__sev_value_sub",
                BinaryOp::Mul => "__sev_value_mul",
                BinaryOp::Div => "__sev_value_div",
                _ => unreachable!(),
            };
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @{function}({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
            return (result, ValueType::Any);
        }
        if operand_type == ValueType::Any
            && right_type == ValueType::Any
            && matches!(
                op,
                BinaryOp::Equal
                    | BinaryOp::NotEqual
                    | BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual
            )
        {
            let (function, first, second, invert) = match op {
                BinaryOp::Equal => ("__sev_value_equal", left.as_str(), right.as_str(), false),
                BinaryOp::NotEqual => ("__sev_value_equal", left.as_str(), right.as_str(), true),
                BinaryOp::Less => ("__sev_value_less", left.as_str(), right.as_str(), false),
                BinaryOp::LessEqual => ("__sev_value_less", right.as_str(), left.as_str(), true),
                BinaryOp::Greater => ("__sev_value_less", right.as_str(), left.as_str(), false),
                BinaryOp::GreaterEqual => ("__sev_value_less", left.as_str(), right.as_str(), true),
                _ => unreachable!(),
            };
            let compared = self.fresh_value();
            writeln!(self.output, "    {compared} = llvm.call @{function}({first}, {second}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            if !invert {
                return (compared, ValueType::Bool);
            }
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1").unwrap();
            let result = self.fresh_value();
            writeln!(
                self.output,
                "    {result} = llvm.xor {compared}, {one} : i1"
            )
            .unwrap();
            return (result, ValueType::Bool);
        }
        if operand_type == ValueType::Any && right_type != ValueType::Any {
            (left, operand_type) = self.unbox_value((left, operand_type), right_type);
        } else if right_type == ValueType::Any && operand_type != ValueType::Any {
            right = self.unbox_value((right, right_type), operand_type).0;
            right_type = operand_type;
        }
        if matches!(op, BinaryOp::Equal | BinaryOp::NotEqual)
            && matches!(
                operand_type,
                ValueType::List | ValueType::Tuple | ValueType::Map | ValueType::Set
            )
            && operand_type == right_type
        {
            let equal = self.fresh_value();
            writeln!(self.output, "    {equal} = llvm.call @__sev_collection_equal({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            if op == BinaryOp::Equal {
                return (equal, ValueType::Bool);
            }
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1").unwrap();
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.xor {equal}, {one} : i1").unwrap();
            return (result, ValueType::Bool);
        }
        if operand_type == ValueType::String && right_type == ValueType::String {
            if op == BinaryOp::Add {
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_string_concat({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                return (result, ValueType::String);
            }
            if matches!(op, BinaryOp::Equal | BinaryOp::NotEqual) {
                let equal = self.fresh_value();
                writeln!(self.output, "    {equal} = llvm.call @__sev_string_equal({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                if op == BinaryOp::Equal {
                    return (equal, ValueType::Bool);
                }
                let one = self.fresh_value();
                writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1").unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.xor {equal}, {one} : i1").unwrap();
                return (result, ValueType::Bool);
            }
        }
        if matches!(op, BinaryOp::Div | BinaryOp::Mod)
            && matches!(operand_type, ValueType::Int | ValueType::Float)
        {
            let zero = self.fresh_value();
            let nonzero = self.fresh_value();
            if operand_type == ValueType::Float {
                writeln!(
                    self.output,
                    "    {zero} = llvm.mlir.constant(0.0 : f64) : f64"
                )
                .unwrap();
                writeln!(
                    self.output,
                    "    {nonzero} = llvm.fcmp \"une\" {right}, {zero} : f64"
                )
                .unwrap();
            } else {
                writeln!(
                    self.output,
                    "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                writeln!(
                    self.output,
                    "    {nonzero} = llvm.icmp \"ne\" {right}, {zero} : i64"
                )
                .unwrap();
            }
            let valid = self.fresh_block();
            let failed = self.fresh_block();
            writeln!(
                self.output,
                "    llvm.cond_br {nonzero}, ^bb{valid}, ^bb{failed}"
            )
            .unwrap();
            writeln!(self.output, "  ^bb{failed}:").unwrap();
            writeln!(
                self.output,
                "    llvm.call @__sev_runtime_fail_division_zero() : () -> ()"
            )
            .unwrap();
            writeln!(self.output, "    llvm.unreachable").unwrap();
            writeln!(self.output, "  ^bb{valid}:").unwrap();
        }
        let result = self.fresh_value();
        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            let operation = if op == BinaryOp::And {
                "llvm.and"
            } else {
                "llvm.or"
            };
            writeln!(
                self.output,
                "    {result} = {operation} {left}, {right} : i1"
            )
            .unwrap();
            return (result, ValueType::Bool);
        }
        let (operation, result_type) = match op {
            BinaryOp::Add => (
                if operand_type == ValueType::Float {
                    "llvm.fadd"
                } else {
                    "llvm.add"
                },
                operand_type,
            ),
            BinaryOp::Sub => (
                if operand_type == ValueType::Float {
                    "llvm.fsub"
                } else {
                    "llvm.sub"
                },
                operand_type,
            ),
            BinaryOp::Mul => (
                if operand_type == ValueType::Float {
                    "llvm.fmul"
                } else {
                    "llvm.mul"
                },
                operand_type,
            ),
            BinaryOp::Div => (
                if operand_type == ValueType::Float {
                    "llvm.fdiv"
                } else {
                    "llvm.sdiv"
                },
                operand_type,
            ),
            BinaryOp::Mod => (
                if operand_type == ValueType::Float {
                    "llvm.frem"
                } else {
                    "llvm.srem"
                },
                operand_type,
            ),
            comparison => {
                let predicate = match comparison {
                    BinaryOp::Equal => "eq",
                    BinaryOp::NotEqual => "ne",
                    BinaryOp::Less => "slt",
                    BinaryOp::LessEqual => "sle",
                    BinaryOp::Greater => "sgt",
                    BinaryOp::GreaterEqual => "sge",
                    BinaryOp::In => {
                        writeln!(
                            self.output,
                            "    {result} = llvm.mlir.constant(0 : i1) : i1"
                        )
                        .unwrap();
                        return (result, ValueType::Bool);
                    }
                    _ => unreachable!(),
                };
                if operand_type == ValueType::Float {
                    let float_predicate = match comparison {
                        BinaryOp::Equal => "oeq",
                        BinaryOp::NotEqual => "one",
                        BinaryOp::Less => "olt",
                        BinaryOp::LessEqual => "ole",
                        BinaryOp::Greater => "ogt",
                        BinaryOp::GreaterEqual => "oge",
                        _ => unreachable!(),
                    };
                    writeln!(
                        self.output,
                        "    {result} = llvm.fcmp \"{float_predicate}\" {left}, {right} : f64"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        self.output,
                        "    {result} = llvm.icmp \"{predicate}\" {left}, {right} : {}",
                        mlir_type(operand_type)
                    )
                    .unwrap();
                }
                return (result, ValueType::Bool);
            }
        };
        writeln!(
            self.output,
            "    {result} = {operation} {left}, {right} : {}",
            mlir_type(operand_type)
        )
        .unwrap();
        (result, result_type)
    }

    pub(super) fn lower_power_values(
        &mut self,
        (mut base, base_type): (String, ValueType),
        (exponent, exponent_type): (String, ValueType),
    ) -> (String, ValueType) {
        let base_type = if base_type == ValueType::Any {
            let unboxed = self.unbox_value((base, base_type), ValueType::Float);
            base = unboxed.0;
            ValueType::Float
        } else {
            base_type
        };
        if !matches!(base_type, ValueType::Int | ValueType::Float)
            || !matches!(exponent_type, ValueType::Int | ValueType::Float)
        {
            let result = self.fresh_value();
            writeln!(
                self.output,
                "    {result} = llvm.mlir.constant(0.0 : f64) : f64"
            )
            .unwrap();
            return (result, ValueType::Any);
        }

        if base_type == ValueType::Int {
            let converted = self.fresh_value();
            writeln!(
                self.output,
                "    {converted} = llvm.sitofp {base} : i64 to f64"
            )
            .unwrap();
            base = converted;
        }

        let powered = self.fresh_value();
        if exponent_type == ValueType::Int {
            writeln!(
                self.output,
                "    {powered} = llvm.intr.powi({base}, {exponent}) : (f64, i64) -> f64"
            )
            .unwrap();
        } else {
            writeln!(
                self.output,
                "    {powered} = llvm.intr.pow({base}, {exponent}) : (f64, f64) -> f64"
            )
            .unwrap();
        }

        if base_type == ValueType::Int && exponent_type == ValueType::Int {
            let result = self.fresh_value();
            writeln!(
                self.output,
                "    {result} = llvm.fptosi {powered} : f64 to i64"
            )
            .unwrap();
            (result, ValueType::Int)
        } else {
            (powered, ValueType::Float)
        }
    }

    pub(super) fn fresh_value(&mut self) -> String {
        let value = format!("%v{}", self.next_value);
        self.next_value += 1;
        value
    }

    pub(super) fn lower_formatted_print(&mut self, format: &str, value: &str, ty: ValueType) {
        let format_value = self.fresh_value();
        writeln!(
            self.output,
            "    {format_value} = llvm.mlir.addressof {format} : !llvm.ptr"
        )
        .unwrap();
        let status = self.fresh_value();
        writeln!(
            self.output,
            "    {status} = llvm.call @printf({format_value}, {value}) vararg(!llvm.func<i32 (!llvm.ptr, ...)>) : (!llvm.ptr, {}) -> i32",
            mlir_type(ty)
        )
        .unwrap();
    }
}
