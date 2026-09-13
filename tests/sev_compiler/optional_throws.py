#!/usr/bin/env python3
"""Optional fallback throws must execute and expected throws must be checked."""
import unittest

from migration import COMPILER, MigrationCase, ROOT


class OptionalThrows(MigrationCase):
    def test_documented_optionals(self):
        path = ROOT / 'docs/examples/01-types/01-basic/08-optionals.sev'
        self.succeeds([COMPILER, 'test', path, '--sysroot', ROOT,
                       '-o', self.directory / 'optionals'])

    def test_error_message_and_fallible_call(self):
        self.native('''
            def checked() -> int | MessageError:
                throw Error("missing")
            test:
                value = Error("message")
                assert(value.message == "message")
                failure ?= checked()
                if failure is MessageError:
                    assert(failure.message == "missing")
                else:
                    assert(false)
                throws(checked())
                print("continued")
        ''', expected='continued\n')

    def test_fallback_evaluates_once_and_only_throws_when_absent(self):
        self.native('''
            def maybe(found: bool) -> string | None:
                print("lookup")
                if found:
                    return "value"
                return None
            def message() -> string:
                print("message")
                return "missing"
            test:
                present = maybe(true) else throw Error(message())
                assert(present == "value")
                throws(maybe(false) else throw Error(message()))
                print("continued")
        ''', expected='lookup\nlookup\nmessage\ncontinued\n')

    def test_no_throw_fails_at_runtime(self):
        for expression in ['42', 'maybe(true) else throw Error("missing")']:
            with self.subTest(expression=expression):
                path = self.write('''
                    def maybe(found: bool) -> string | None:
                        if found:
                            return "present"
                        return None
                    test:
                        throws(''' + expression + ''')
                        print("must not continue")
                ''')
                result = self.invoke([COMPILER, 'test', path, '--sysroot', ROOT,
                                      '-o', self.directory / 'no-throw'])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('expected an error to be thrown', result.stderr)
                self.assertNotIn('must not continue', result.stdout)

    def test_uncaught_throw_still_fails(self):
        path = self.write('''
            def main():
                throw Error("uncaught")
            test:
                main()
        ''')
        self.compile(path)
        result = self.invoke([COMPILER, 'test', path, '--sysroot', ROOT,
                              '-o', self.directory / 'uncaught'])
        self.assertNotEqual(result.returncode, 0)

    def test_throws_does_not_accept_compiler_errors(self):
        self.rejects('test:\n    throws(missing_callable())',
                     'unknown callable missing_callable')
        self.rejects('test:\n    throws(throw 42)',
                     'throw requires a value conforming to Error')
        self.rejects('test:\n    throws()',
                     'throws requires exactly one positional expression')


if __name__ == '__main__':
    unittest.main()
