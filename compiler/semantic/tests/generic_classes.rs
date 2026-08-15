use severian_hir::{Instruction, TensorElementType, TypeDefinitionId, ValueType};
use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::{analyze, analyze_with_interfaces};

fn analyze_source(source: &str) -> Result<severian_hir::Program, severian_semantic::SemanticError> {
    analyze(&parse(&lex(source).unwrap()).unwrap())
}

const PRELUDE: &str = concat!(
    "trait TensorDType:\n",
    "    zero() -> Self\n",
    "\n",
    "trait Module[T: TensorDType]:\n",
    "    def forward(x: Tensor[T]) -> Tensor[T]\n",
    "\n",
    "class Linear[T: TensorDType]:\n",
    "    weight: Tensor[T]\n",
    "\n",
    "    def forward(x: Tensor[T]) -> Tensor[T]:\n",
    "        return x\n",
    "\n",
);

#[test]
fn specializes_generic_classes_for_each_concrete_type() {
    let source = format!(
        "{}{}",
        PRELUDE,
        concat!(
            "def use_f32(weight: Tensor[f32], x: Tensor[f32]) -> Tensor[f32]:\n",
            "    layer: Linear[f32] = Linear[f32](weight)\n",
            "    return layer.forward(x)\n",
            "\n",
            "def use_bf16(weight: Tensor[bf16], x: Tensor[bf16]) -> Tensor[bf16]:\n",
            "    layer: Linear[bf16] = Linear[bf16](weight)\n",
            "    return layer.forward(x)\n",
        )
    );
    let program = analyze_source(&source).unwrap();

    assert!(program.classes.iter().all(|class| class.name != "Linear"));
    for (name, element) in [
        ("Linear__f32", TensorElementType::F32),
        ("Linear__bf16", TensorElementType::BF16),
    ] {
        let class = program
            .classes
            .iter()
            .find(|class| class.name == name)
            .unwrap_or_else(|| panic!("missing specialization {name}"));
        assert!(matches!(
            class.field_types[0],
            ValueType::Tensor(tensor) if tensor.element == element
        ));
    }
}

#[test]
fn specializes_classes_with_multiple_type_arguments() {
    let source = concat!(
        "class Pair[Left, Right]:\n",
        "    left: Left\n",
        "    right: Right\n",
        "\n",
        "def pair() -> Pair[int, string]:\n",
        "    return Pair[int, string](1, \"one\")\n",
    );
    let program = analyze_source(source).unwrap();
    let pair = program
        .classes
        .iter()
        .find(|class| class.name == "Pair__int__string")
        .expect("missing two-argument specialization");
    assert_eq!(pair.field_types, [ValueType::Int, ValueType::String]);
}

#[test]
fn transition_states_gate_typestate_methods_during_specialization() {
    let source = concat!(
        "enum SocketState:\n",
        "    Closed -> Connected\n",
        "    Connected -> Closed\n",
        "\n",
        "class Socket[State]:\n",
        "    descriptor: int\n",
        "\n",
        "    def connect() -> Socket[Connected] with { State == Closed }:\n",
        "        return Socket[Connected](descriptor)\n",
        "\n",
        "    def send(data: string) -> int with { State == Connected }:\n",
        "        return size(data)\n",
        "\n",
        "def transmit(socket: Socket[Closed]) -> int:\n",
        "    connected = socket.connect()\n",
        "    return connected.send(\"hello\")\n",
    );
    let program = analyze_source(source).unwrap();
    let closed = program
        .classes
        .iter()
        .find(|class| class.name == "Socket__Closed")
        .unwrap();
    let connected = program
        .classes
        .iter()
        .find(|class| class.name == "Socket__Connected")
        .unwrap();
    assert_eq!(
        closed
            .methods
            .iter()
            .map(|method| method.name.as_str())
            .collect::<Vec<_>>(),
        ["connect"]
    );
    assert_eq!(
        connected
            .methods
            .iter()
            .map(|method| method.name.as_str())
            .collect::<Vec<_>>(),
        ["send"]
    );
}

#[test]
fn rejects_a_method_unavailable_in_the_current_typestate() {
    let source = concat!(
        "enum SocketState:\n",
        "    Closed -> Connected\n",
        "    Connected\n",
        "\n",
        "class Socket[State]:\n",
        "    def send(data: string) with { State == Connected }:\n",
        "        return\n",
        "\n",
        "def invalid(socket: Socket[Closed]):\n",
        "    socket.send(\"too early\")\n",
    );
    let error = analyze_source(source).unwrap_err();
    assert!(error.message.contains("method `send` is not available"));
}

#[test]
fn typestate_rebinding_advances_the_receiver_type() {
    let source = concat!(
        "enum SocketState:\n",
        "    Closed -> Connected\n",
        "    Connected -> Closed\n",
        "\n",
        "class Socket[State]:\n",
        "    descriptor: int\n",
        "\n",
        "    def connect() -> Socket[Connected] with { State == Closed }:\n",
        "        return Socket[Connected](descriptor)\n",
        "\n",
        "    def send(data: string) -> int with { State == Connected }:\n",
        "        return size(data)\n",
        "\n",
        "def transmit(socket: Socket[Closed]) -> int:\n",
        "    current := socket\n",
        "    current = current.connect()\n",
        "    return current.send(\"hello\")\n",
    );
    analyze_source(source).unwrap();
}

#[test]
fn rejects_an_edge_missing_from_the_typestate_graph() {
    let source = concat!(
        "enum DownloadState:\n",
        "    Pending -> Connecting\n",
        "    Connecting -> Complete\n",
        "    Complete\n",
        "\n",
        "class Download[State]:\n",
        "    passcode: int\n",
        "\n",
        "def skip(download: Download[Pending]):\n",
        "    current := download\n",
        "    current = Download[Complete](0)\n",
    );
    let error = analyze_source(source).unwrap_err();
    assert!(error.message.contains("invalid typestate transition"));
}

#[test]
fn accepts_structural_generic_trait_conformance() {
    let source = format!(
        "{}{}",
        PRELUDE,
        concat!(
            "def consume(module: Module[f32], x: Tensor[f32]) -> Tensor[f32]:\n",
            "    return module.forward(x)\n",
            "\n",
            "def use_linear(weight: Tensor[f32], x: Tensor[f32]) -> Tensor[f32]:\n",
            "    layer = Linear[f32](weight)\n",
            "    return consume(layer, x)\n",
        )
    );

    let program = analyze_source(&source).unwrap();
    let consume = program
        .functions
        .iter()
        .find(|function| function.name == "consume")
        .unwrap();
    let interface = ValueType::Interface(TypeDefinitionId::from_name("Module[f32]"));
    assert_eq!(consume.params[0].ty, interface);
    assert_ne!(consume.params[0].ty, ValueType::Any);
}

#[test]
fn preserves_generic_interface_identity_through_result_propagation() {
    let source = format!(
        "{}{}",
        PRELUDE,
        concat!(
            "def load(weight: Tensor[f32]) -> Result[Module[f32], string]:\n",
            "    return Linear[f32](weight)\n",
            "\n",
            "def apply(weight: Tensor[f32], x: Tensor[f32]) -> Result[Tensor[f32], string]:\n",
            "    module = load(weight)\n",
            "    return module.forward(x)\n",
        )
    );
    let program = analyze_source(&source).unwrap();
    let apply = program
        .functions
        .iter()
        .find(|function| function.name == "apply")
        .unwrap();
    let Instruction::TryLet {
        payload_type,
        receiver: Some(receiver),
        ..
    } = &apply.instructions[0]
    else {
        panic!("expected a typed propagated Result payload");
    };
    assert_eq!(
        *payload_type,
        ValueType::Interface(TypeDefinitionId::from_name("Module[f32]"))
    );
    assert_eq!(receiver.name, "Module[f32]");
    assert!(!receiver.concrete);
}

#[test]
fn keeps_generic_interface_specializations_distinct() {
    let source = format!(
        "{}{}",
        PRELUDE,
        concat!(
            "def use_f32(module: Module[f32], x: Tensor[f32]) -> Tensor[f32]:\n",
            "    return module.forward(x)\n",
            "\n",
            "def use_bf16(module: Module[bf16], x: Tensor[bf16]) -> Tensor[bf16]:\n",
            "    return module.forward(x)\n",
        )
    );
    let program = analyze_source(&source).unwrap();
    let f32_interface = program.functions[0].params[0].ty;
    let bf16_interface = program.functions[1].params[0].ty;
    assert_ne!(f32_interface, bf16_interface);
    assert!(matches!(f32_interface, ValueType::Interface(_)));
    assert!(matches!(bf16_interface, ValueType::Interface(_)));
}

#[test]
fn rejects_a_structural_trait_signature_mismatch() {
    let source = format!(
        "{}{}",
        PRELUDE,
        concat!(
            "class Broken[T: TensorDType]:\n",
            "    weight: Tensor[T]\n",
            "\n",
            "    def forward(x: Tensor[T]) -> int:\n",
            "        return 1\n",
            "\n",
            "def consume(module: Module[f32], x: Tensor[f32]) -> Tensor[f32]:\n",
            "    return module.forward(x)\n",
            "\n",
            "def use_broken(weight: Tensor[f32], x: Tensor[f32]) -> Tensor[f32]:\n",
            "    broken = Broken[f32](weight)\n",
            "    return consume(broken, x)\n",
        )
    );

    let error = analyze_source(&source).unwrap_err();
    assert!(error
        .message
        .contains("does not structurally satisfy `Module[f32]`"));
}

#[test]
fn rejects_a_type_argument_outside_its_bound() {
    let source = concat!(
        "class FloatBox[T: Float]:\n",
        "    value: T\n",
        "\n",
        "def invalid(value: i64):\n",
        "    box = FloatBox[i64](value)\n",
    );

    let error = analyze_source(source).unwrap_err();
    assert!(error.message.contains("does not satisfy `Float`"));
}

#[test]
fn class_bounds_use_structural_trait_conformance() {
    let source = concat!(
        "trait Serializable:\n",
        "    encode() -> string\n",
        "\n",
        "class Record:\n",
        "    value: string\n",
        "\n",
        "    def encode() -> string:\n",
        "        return value\n",
        "\n",
        "class Envelope[T: Serializable]:\n",
        "    value: T\n",
        "\n",
        "def wrap(value: Record) -> Envelope[Record]:\n",
        "    return Envelope[Record](value)\n",
    );

    let program = analyze_source(source).unwrap();
    assert!(program
        .classes
        .iter()
        .any(|class| class.name == "Envelope__Record"));
}

#[test]
fn imported_generic_traits_retain_their_concrete_method_contract() {
    let contracts = parse(
        &lex(concat!(
            "trait Module[T: TensorDType]:\n",
            "    def forward(x: Tensor[T]) -> Tensor[T]\n",
        ))
        .unwrap(),
    )
    .unwrap();
    let source = concat!(
        "import contracts\n",
        "\n",
        "class Linear:\n",
        "    weight: Tensor[f32]\n",
        "\n",
        "    def forward(x: Tensor[f32]) -> Tensor[f32]:\n",
        "        return x\n",
        "\n",
        "def consume(module: contracts.Module[f32], x: Tensor[f32]) -> Tensor[f32]:\n",
        "    return module.forward(x)\n",
        "\n",
        "def apply(weight: Tensor[f32], x: Tensor[f32]) -> Tensor[f32]:\n",
        "    return consume(Linear(weight), x)\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();

    let program = analyze_with_interfaces(&module, &[("contracts".into(), contracts)]).unwrap();
    let consume = program
        .functions
        .iter()
        .find(|function| function.name == "consume")
        .unwrap();
    assert_eq!(
        consume.params[0].ty,
        ValueType::Interface(TypeDefinitionId::from_name("contracts.Module[f32]"))
    );
}

#[test]
fn imported_generic_classes_specialize_in_the_consuming_module() {
    let boxes = parse(
        &lex(concat!(
            "class Box[T: Numeric]:\n",
            "    value: T\n",
            "\n",
            "    def get() -> T:\n",
            "        return value\n",
        ))
        .unwrap(),
    )
    .unwrap();
    let source = concat!(
        "import boxes\n",
        "\n",
        "def read(value: f32) -> f32:\n",
        "    boxed = boxes.Box[f32](value)\n",
        "    return boxed.get()\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();

    let program = analyze_with_interfaces(&module, &[("boxes".into(), boxes)]).unwrap();
    assert!(
        program
            .classes
            .iter()
            .any(|class| class.name == "boxes_Box__f32"),
        "classes: {:?}",
        program
            .classes
            .iter()
            .map(|class| &class.name)
            .collect::<Vec<_>>()
    );
}

#[test]
fn imported_generic_classes_specialize_nested_sibling_fields() {
    let collections = parse(
        &lex(concat!(
            "class Deque[T]:\n",
            "    values: list[T]\n",
            "\n",
            "class Queue[T]:\n",
            "    deque: Deque[T]\n",
        ))
        .unwrap(),
    )
    .unwrap();
    let source = concat!(
        "import collections\n",
        "\n",
        "def queue() -> collections.Queue[int]:\n",
        "    return collections.Queue[int](collections.Deque[int]([]))\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let program = analyze_with_interfaces(&module, &[("collections".into(), collections)]).unwrap();
    let queue = program
        .classes
        .iter()
        .find(|class| class.name == "collections_Queue__int")
        .expect("missing imported queue specialization");
    assert_eq!(
        queue.field_classes[0].as_deref(),
        Some("collections_Deque__int")
    );
    assert!(program
        .classes
        .iter()
        .any(|class| class.name == "collections_Deque__int"));
}
