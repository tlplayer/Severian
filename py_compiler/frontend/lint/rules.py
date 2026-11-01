"""Frontend lint contracts shared by general policy and syntax Y/G providers."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json


@dataclass(frozen=True)
class SourceEdit:
    span: object
    replacement: str
    rule: str


@dataclass(frozen=True)
class LintFinding:
    span: object
    message: str
    replacement: str | None = None


@dataclass(frozen=True)
class LintRule:
    name: str
    phase: str
    check: object = None

    def __post_init__(self):
        if self.phase not in ('source', 'syntax', 'resolved'):
            raise ValueError('unknown lint phase ' + self.phase)
        if self.check is not None and not callable(self.check):
            raise ValueError('lint rule check must be callable')


@lru_cache(maxsize=1)
def policy():
    result = json.loads(Path(__file__).with_name('sev.grammar').read_text(encoding='utf-8'))
    if result.get('version') != 1:
        raise ValueError('unsupported frontend lint grammar version')
    for key in ('columns', 'indentation', 'conditions', 'operation_block', 'positional_arguments', 'maximum_passes'):
        if type(result.get(key)) is not int or result[key] < 1:
            raise ValueError('lint grammar requires a positive ' + key)
    if not all(isinstance(key, str) and isinstance(value, str)
               for key, value in result.get('abbreviations', {}).items()):
        raise ValueError('lint abbreviation maps require string spellings')
    return result


def apply_edits(source, edits):
    """Every edit belongs to this immutable snapshot; conflicting fixes fail."""
    from py_compiler.frontend.source.source import Diagnostic
    unique = {}
    for edit in edits:
        if edit.span.source != source.identity:
            raise ValueError('lint fix belongs to another source snapshot')
        source.span(edit.span.start, edit.span.end)
        key = (edit.span.start, edit.span.end)
        if key in unique and unique[key].replacement != edit.replacement:
            raise Diagnostic('lint', 'conflicting fixes for ' + edit.rule, source, edit.span)
        unique[key] = edit
    ordered = sorted(unique.values(), key=lambda edit: (edit.span.start, edit.span.end))
    stop = 0
    for edit in ordered:
        if edit.span.start < stop:
            raise Diagnostic('lint', 'overlapping fixes for ' + edit.rule, source, edit.span)
        stop = edit.span.end
    text = source.text
    for edit in reversed(ordered):
        text = text[:edit.span.start] + edit.replacement + text[edit.span.end:]
    return text


def declared_rules(syntax):
    """The active registry supplies local Y/G rules, including extensions."""
    found = {}
    for provider in syntax.registry.providers:
        for rule in getattr(provider, 'lint_rules', lambda: ())():
            previous = found.get(rule.name)
            if previous is not None and previous != rule:
                raise ValueError('conflicting syntax lint rule ' + rule.name)
            found[rule.name] = rule
    return tuple(found.values())


import unittest


class LintContractTests(unittest.TestCase):
    def test_conflicting_and_stale_fixes_are_rejected(self):
        from py_compiler.frontend.source.source import SourceFile, Diagnostic
        source = SourceFile('fix.sev', 'abc')
        self.assertEqual(apply_edits(source, [SourceEdit(source.span(0, 1), 'A', 'case')]), 'Abc')
        with self.assertRaises(Diagnostic):
            apply_edits(source, [SourceEdit(source.span(0, 2), 'A', 'one'),
                                 SourceEdit(source.span(1, 3), 'B', 'two')])
        with self.assertRaises(ValueError):
            apply_edits(source, [SourceEdit(SourceFile('other.sev', 'abc').span(0, 1), 'A', 'case')])
