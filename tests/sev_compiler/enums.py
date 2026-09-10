#!/usr/bin/env python3
"""Native and diagnostic regressions for resolved enum/union patterns."""
import unittest
from migration import MigrationCase, ROOT

SHAPES = '''
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)
    Trapezoid(side: float, width: float, height: float)
    Oblong(height: float, width: float)
    Point
'''


class EnumPatterns(MigrationCase):
    def test_documented_examples(self):
        for path in sorted((ROOT / 'docs/examples/01-types/05-enums').glob('*.sev')):
            with self.subTest(path=path.name):
                self.native(path.read_text())

    def test_all_pattern_forms_and_different_field_positions(self):
        self.native(SHAPES + '''
def area(shape: Shape) -> float:
    return match shape:
        case Circle(r):
            r * r
        case rectangle: Rectangle:
            rectangle.width * rectangle.height
        case quad: Trapezoid | Oblong:
            quad.width * quad.height
        case Point:
            0.0
def tag_only(shape: Shape) -> int:
    match shape:
        case Circle:
            return 1
        case _:
            return 0
test:
    assert(area(Circle(2.0)) == 4.0)
    assert(area(Rectangle(3.0, 4.0)) == 12.0)
    assert(area(Trapezoid(9.0, 3.0, 4.0)) == 12.0)
    assert(area(Oblong(4.0, 3.0)) == 12.0)
    assert(area(Point) == 0.0)
    assert(tag_only(Circle(2.0)) == 1)
''')

    def test_value_match_evaluates_subject_once_and_selected_arm_only(self):
        self.native('''
class Counter:
    count: int
    def take() -> int:
        count += 1
        return count
test:
    counter = Counter(0)
    value = match counter.take():
        case 1:
            base = 40
            base + 2
        case _:
            counter.take()
    assert(value == 42)
    assert(counter.count == 1)
''')

    def test_general_union_and_nested_narrowing(self):
        self.native('''
def read(value: int | string) -> int:
    return match value:
        case number: int:
            number + 1
        case text: string:
            4 if text == "four" else 0
test:
    assert(read(41) == 42)
    assert(read("four") == 4)
''')
        self.native(SHAPES + '''
def read(shape: Shape) -> float:
    return match shape:
        case quad: Trapezoid | Oblong:
            match quad:
                case trap: Trapezoid:
                    trap.side
                case oblong: Oblong:
                    oblong.width
        case _:
            0.0
test:
    assert(read(Trapezoid(42.0, 3.0, 4.0)) == 42.0)
    assert(read(Oblong(4.0, 3.0)) == 3.0)
''')

    def test_common_member_types_remain_a_union(self):
        self.native("""
enum Number:
    Small(value: i32)
    Large(value: int)
def read(number: Number) -> int:
    return match number:
        case item: Small | Large:
            match item.value:
                case small: i32:
                    int(small)
                case large: int:
                    large
test:
    assert(read(Small(i32(42))) == 42)
    assert(read(Large(43)) == 43)
""")

    def test_capture_shadows_subject_only_inside_arm(self):
        self.native(SHAPES + """
def read(shape: Shape) -> float:
    result = match shape:
        case shape: Circle:
            shape.radius
        case _:
            0.0
    match shape:
        case Circle(_):
            return result
        case _:
            return 0.0
test:
    assert(read(Circle(42.0)) == 42.0)
""")

    def test_explicit_variant_type(self):
        self.native(SHAPES + '''
def radius(circle: Shape.Circle) -> float:
    return circle.radius
def read(shape: Shape) -> float:
    return match shape:
        case circle: Circle:
            radius(circle)
        case _:
            0.0
test:
    circle: Shape.Circle = Circle(2.0)
    assert(radius(circle) == 2.0)
    assert(read(circle) == 2.0)
    assert(read(Circle(3.0)) == 3.0)
''')

    def test_invalid_patterns_and_members(self):
        cases = [
            ('case Circle:\n            return radius', 'unknown name radius'),
            ('case Circle(a, b):\n            return 0.0', 'wrong payload arity'),
            ('case Rectangle(a, a):\n            return a', 'duplicate payload binding'),
            ('case Circle(1):\n            return 0.0', 'payload pattern requires'),
            ('case quad: Trapezoid | Oblong:\n            return quad.side', 'not available on every member'),
            ('case value: int:\n            return 0.0', 'not a variant'),
            ('case Circle:\n            return 0.0\n        case Circle(r):\n            return r', 'unreachable tagged match arm'),
        ]
        for arm, diagnostic in cases:
            with self.subTest(arm=arm):
                self.rejects(SHAPES + '\ndef read(shape: Shape) -> float:\n    match shape:\n        ' + arm + '\n        case _:\n            return 0.0\n', diagnostic)

    def test_exhaustiveness_and_result_types(self):
        self.rejects(SHAPES + '''
def read(shape: Shape) -> float:
    return match shape:
        case Circle(r):
            r
''', 'non-exhaustive tagged match')
        self.rejects('''
def read(value: int) -> int:
    return match value:
        case 1:
            42
''', 'requires exhaustive')
        self.rejects('''
def read(value: int) -> int:
    return match value:
        case 1:
            42
        case _:
            "wrong"
''', 'expected scalar type')

    def test_transition_graph(self):
        graph = '''
enum Status:
    Connecting -> Received | Failed
    Received
    Failed
'''
        self.native(graph + '''
test:
    state := Connecting
    state = Received
''')
        self.rejects(graph + '''
def invalid():
    state := Received
    state = Connecting
''', 'invalid state transition Received -> Connecting')
        self.rejects(graph + '''
def invalid(flag: bool):
    state := Connecting
    if flag:
        state = Received
    state = Failed
''', 'invalid state transition Received -> Failed')
        self.rejects('enum State:\n    First -> Missing\n', 'unknown transition destination')
        self.rejects('enum State:\n    First -> Last | Last\n    Last\n', 'duplicate transition destination')


if __name__ == '__main__':
    unittest.main(verbosity=2)
