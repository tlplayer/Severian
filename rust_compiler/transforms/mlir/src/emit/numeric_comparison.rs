use super::*;

/// Compare numeric values without first converting either into the other's
/// domain. Temporary names are scoped by the result's unique SSA identity.
pub(super) fn render_numeric_comparison(
    output: &mut String,
    module: &Module,
    operator: BinaryOperation,
    left: ValueId,
    right: ValueId,
    result: ValueId,
    indent: usize,
) -> Result<bool, MlirError> {
    if !matches!(operator, BinaryOperation::Equal | BinaryOperation::NotEqual
        | BinaryOperation::Less | BinaryOperation::LessEqual
        | BinaryOperation::Greater | BinaryOperation::GreaterEqual)
    {
        return Ok(false);
    }
    let lt = value_type(module, left)?;
    let rt = value_type(module, right)?;
    if lt == rt {
        return Ok(false);
    }
    let prefix = format!("v{}_compare", result.0);
    let indentation = " ".repeat(indent);
    let mut emit = |name: &str, instruction: String| -> String {
        let name = format!("%{prefix}_{name}");
        output.push_str(&format!("{indentation}{name} = {instruction}\n"));
        name
    };
    let lhs = format!("%v{}", left.0);
    let rhs = format!("%v{}", right.0);
    let instruction = match (&lt, &rt) {
        (LoweredType::Integer { bits: lb, signed: ls }, LoweredType::Integer { bits: rb, signed: rs }) => {
            let signed = *ls || *rs;
            // An unsigned operand needs a spare sign bit when sharing a signed
            // domain, including the i128/u128 pair (which compares in i129).
            let bits = (*lb + u16::from(signed && !ls)).max(*rb + u16::from(signed && !rs));
            let widen = |emit: &mut dyn FnMut(&str, String) -> String, name, value: String, width, signed| {
                if width == bits { value } else {
                    emit(name, format!("arith.{} {value} : i{width} to i{bits}", if signed { "extsi" } else { "extui" }))
                }
            };
            let lhs = widen(&mut emit, "left", lhs, *lb, *ls);
            let rhs = widen(&mut emit, "right", rhs, *rb, *rs);
            let ty = LoweredType::Integer { bits, signed };
            format!("{} {lhs}, {rhs} : i{bits}", binary_mnemonic(operator, &ty)?)
        }
        (LoweredType::Float { format: lf }, LoweredType::Float { format: rf }) => {
            // f32 contains both f8 formats, f16 and bf16 exactly.
            let width = |format| match format { LoweredFloatFormat::Ieee(bits) => bits.max(32), _ => 32 };
            let bits = width(*lf).max(width(*rf));
            let ty = LoweredType::Float { format: LoweredFloatFormat::Ieee(bits) };
            let spelling = mlir_type(&ty)?;
            let lhs = if lt == ty { lhs } else { emit("left", format!("arith.extf {lhs} : {} to {spelling}", mlir_type(&lt)?)) };
            let rhs = if rt == ty { rhs } else { emit("right", format!("arith.extf {rhs} : {} to {spelling}", mlir_type(&rt)?)) };
            format!("{} {lhs}, {rhs} : {spelling}", binary_mnemonic(operator, &ty)?)
        }
        (LoweredType::Integer { .. }, LoweredType::Float { .. })
        | (LoweredType::Float { .. }, LoweredType::Integer { .. }) => {
            let (integer, floating, integer_type, float_type, reverse) = if matches!(lt, LoweredType::Integer { .. }) {
                (lhs, rhs, &lt, &rt, false)
            } else { (rhs, lhs, &rt, &lt, true) };
            let LoweredType::Integer { bits, signed } = integer_type else { unreachable!() };
            let wide_float = match float_type {
                LoweredType::Float { format: LoweredFloatFormat::Ieee(128) } => "f128",
                _ => "f64",
            };
            let integer = if *bits == 128 { integer } else {
                emit("integer", format!("arith.{} {integer} : i{bits} to i128", if *signed { "extsi" } else { "extui" }))
            };
            let floating = if mlir_type(float_type)? == wide_float { floating } else {
                emit("float", format!("arith.extf {floating} : {} to {wide_float}", mlir_type(float_type)?))
            };
            let zero = emit("zero", "arith.constant 0 : i128".into());
            let float_zero = emit("float_zero", format!("arith.constant 0.0 : {wide_float}"));
            let int_negative = if *signed {
                emit("int_negative", format!("arith.cmpi slt, {integer}, {zero} : i128"))
            } else { emit("int_negative", "arith.constant false".into()) };
            let float_negative = emit("float_negative", format!("arith.cmpf olt, {floating}, {float_zero} : {wide_float}"));
            let negated = emit("negated", format!("arith.subi {zero}, {integer} : i128"));
            let magnitude = emit("magnitude", format!("arith.select {int_negative}, {negated}, {integer} : i128"));
            let float_negated = emit("float_negated", format!("arith.negf {floating} : {wide_float}"));
            let float_magnitude = emit("float_magnitude", format!("arith.select {float_negative}, {float_negated}, {floating} : {wide_float}"));
            let upper = emit("upper", format!("arith.constant 340282366920938463463374607431768211456.0 : {wide_float}"));
            let in_range = emit("in_range", format!("arith.cmpf olt, {float_magnitude}, {upper} : {wide_float}"));
            // The selected input is always finite and in [0, 2^128). Do not
            // emit an unchecked FP-to-int conversion on NaN or infinity.
            let safe = emit("safe", format!("arith.select {in_range}, {float_magnitude}, {float_zero} : {wide_float}"));
            let truncated = emit("truncated", format!("arith.fptoui {safe} : {wide_float} to i128"));
            let restored = emit("restored", format!("arith.uitofp {truncated} : i128 to {wide_float}"));
            let integral_equal = emit("integral_equal", format!("arith.cmpi eq, {magnitude}, {truncated} : i128"));
            let fractional_equal = emit("fractional_equal", format!("arith.cmpf oeq, {restored}, {float_magnitude} : {wide_float}"));
            let equal = emit("magnitude_equal", format!("arith.andi {integral_equal}, {fractional_equal} : i1"));
            let equal = emit("range_equal", format!("arith.andi {in_range}, {equal} : i1"));
            let fraction = emit("fraction", format!("arith.cmpf olt, {restored}, {float_magnitude} : {wide_float}"));
            let fraction_less = emit("fraction_less", format!("arith.andi {integral_equal}, {fraction} : i1"));
            let integral_less = emit("integral_less", format!("arith.cmpi ult, {magnitude}, {truncated} : i128"));
            let less = emit("magnitude_less", format!("arith.ori {integral_less}, {fraction_less} : i1"));
            let less = emit("range_less", format!("arith.andi {in_range}, {less} : i1"));
            let above = emit("above", format!("arith.cmpf oge, {float_magnitude}, {upper} : {wide_float}"));
            let less = emit("less_or_above", format!("arith.ori {above}, {less} : i1"));
            let greater = emit("integral_greater", format!("arith.cmpi ugt, {magnitude}, {truncated} : i128"));
            let greater = emit("range_greater", format!("arith.andi {in_range}, {greater} : i1"));
            let different = emit("different_signs", format!("arith.xori {int_negative}, {float_negative} : i1"));
            let same = emit("same_signs", format!("arith.cmpi eq, {int_negative}, {float_negative} : i1"));
            let equal = emit("equal", format!("arith.andi {same}, {equal} : i1"));
            let ordered = emit("ordered", format!("arith.cmpf ord, {floating}, {floating} : {wide_float}"));
            let magnitude_less = emit("signed_less", format!("arith.select {int_negative}, {greater}, {less} : i1"));
            let magnitude_greater = emit("signed_greater", format!("arith.select {int_negative}, {less}, {greater} : i1"));
            let less = emit("sign_less", format!("arith.select {different}, {int_negative}, {magnitude_less} : i1"));
            let greater = emit("sign_greater", format!("arith.select {different}, {float_negative}, {magnitude_greater} : i1"));
            let less = emit("less", format!("arith.andi {ordered}, {less} : i1"));
            let greater = emit("greater", format!("arith.andi {ordered}, {greater} : i1"));
            let (less, greater) = if reverse { (greater, less) } else { (less, greater) };
            match operator {
                BinaryOperation::Equal => format!("arith.andi {equal}, {equal} : i1"),
                BinaryOperation::NotEqual => {
                    let one = emit("true", "arith.constant true".into());
                    format!("arith.xori {equal}, {one} : i1")
                }
                BinaryOperation::Less => format!("arith.andi {less}, {less} : i1"),
                BinaryOperation::Greater => format!("arith.andi {greater}, {greater} : i1"),
                BinaryOperation::LessEqual => format!("arith.ori {less}, {equal} : i1"),
                BinaryOperation::GreaterEqual => format!("arith.ori {greater}, {equal} : i1"),
                _ => unreachable!(),
            }
        }
        _ => return Ok(false),
    };
    output.push_str(&format!("{indentation}%v{} = {instruction}\n", result.0));
    Ok(true)
}
