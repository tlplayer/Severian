#!/usr/bin/env python3
"""Executable CFG and compiler-context regressions using the source binary."""
import unittest

from migration import MigrationCase, source


BRANCH = source('''
    trait Guard: G:
        symbol: Y = guard
        arity: int = 2
        left: Operand = Value[bool]
        right: Operand = Body
        control_flow: ControlFlow = Branch
        def semantic(condition: Ex, body: B):
            value = lower(condition)
            taken = cfg.block()
            exit = cfg.block()
            cfg.branch(value, taken, exit)
            cfg.at(taken)
            lower(body)
            cfg.fallthrough(exit)
            cfg.at(exit)
''')


class ExecutableCfg(MigrationCase):
    def test_imported_semantic_branch_edit_changes_execution(self):
        self.write(BRANCH, "control.sev")
        program = '''
            import * from "control.sev"
            test:
                guard true:
                    print("taken")
        '''
        self.native(program, expected="taken\n")
        self.write(BRANCH.replace("cfg.branch(value, taken, exit)",
                                  "cfg.branch(value, exit, taken)"), "control.sev")
        self.native(program)

    def test_terminated_arm_does_not_emit_a_second_terminator(self):
        self.native(BRANCH + source('''
            def answer(value: bool) -> int:
                guard value:
                    return 42
                return 7
            test:
                assert(answer(true) == 42)
                assert(answer(false) == 7)
        '''))

    def test_custom_branch_condition_is_typed(self):
        self.rejects(BRANCH + '\ntest:\n    guard 42:\n        print("bad")\n',
                     r"branch condition requires bool")

    def test_cfg_context_requires_control_flow_capability(self):
        self.rejects(BRANCH.replace("control_flow: ControlFlow = Branch",
                                    "control_flow: ControlFlow | absent = absent") +
                     '\ntest:\n    guard true:\n        print("bad")\n',
                     r"grammar capability")

    def test_block_parameter_argument_count_is_verified(self):
        declaration = BRANCH.replace("cfg.branch(value, taken, exit)",
                                     "parameter = cfg.parameter(exit, value)\n        cfg.branch(value, taken, exit)")
        self.rejects(declaration + '\ntest:\n    guard true:\n        print("bad")\n',
                     r"successor argument count mismatch")

    def test_unterminated_created_block_is_rejected(self):
        declaration = BRANCH.replace("value = lower(condition)",
                                     "unused = cfg.block()\n        value = lower(condition)")
        self.rejects(declaration + '\ntest:\n    guard true:\n        print("bad")\n',
                     r"requires exactly one terminator")

    def test_sibling_value_is_rejected_before_agent_ir_or_mlir(self):
        declaration = BRANCH.replace("lower(body)", "sibling = lower(condition)").replace(
            "cfg.at(exit)", "cfg.at(exit)\n        cfg.branch(sibling, taken, exit)")
        self.rejects(declaration + '\ntest:\n    guard true:\n        print("bad")\n',
                     r"does not dominate its use")

    def test_string_storage_across_loop_and_branch_edges(self):
        self.native('''
            def choose(limit: int) -> string:
                value := "zero"
                i := 0
                while i < limit:
                    i += 1
                    if i == 2:
                        value = "two"
                    else:
                        value = value + "!"
                return value
            test:
                print(choose(0))
                print(choose(3))
        ''', expected="zero\ntwo!\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
