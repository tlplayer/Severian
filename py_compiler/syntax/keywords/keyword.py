"""Keyword declarations shared by recognition and tooling, not lexer conditionals."""

KEYWORDS = frozenset("""
and as async await borrow break case catch class continue copy def drop dynamic
elif else enum false for from global grammar if import in is local match mirror
move not operator or unimplemented return self static test throw trait true try union view
while with yield None absent extend switch unsafe atomic
""".split())
