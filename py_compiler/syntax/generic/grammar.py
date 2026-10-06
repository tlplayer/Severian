"""G: bounded recognition, captures, eligibility and selected construction."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceWindow:
    source: object
    start: int
    end: int
    tokens: tuple = ()

    def __post_init__(self):
        self.source.span(self.start, self.end)
        if any(t.span.source != self.source.identity or t.span.start < self.start or t.span.end > self.end for t in self.tokens):
            raise ValueError('grammar capture escapes its source window')

    @property
    def text(self):
        return self.source.text[self.start:self.end]


@dataclass(frozen=True)
class Match:
    provider: object
    window: SourceWindow
    end: int
    captures: dict = field(default_factory=dict)
    requirements: tuple = ()


class Registry:
    def __init__(self, providers):
        self.providers = tuple(providers)
        for provider in self.providers:
            if not all(callable(getattr(provider, name, None)) for name in ('recognize', 'construct', 'accept')):
                raise ValueError('incomplete grammar provider')

    def recognize(self, role, window, optional=False):
        matches = []
        for provider in self.providers:
            if provider.role != role:
                continue
            match = provider.recognize(window)
            if match is None:
                continue
            if match.window is not window or not window.start < match.end <= window.end:
                raise ValueError('grammar must make progress within the original source window')
            if match.requirements:
                raise ValueError('grammar requirements need capture resolution before construction')
            if provider.accept(match):
                matches.append(match)
        if not matches:
            if optional:
                return None
            raise ValueError(f'no grammar satisfies {role}')
        if len(matches) != 1:
            raise ValueError(f'ambiguous {role} grammar: ' + ', '.join(type(m.provider).__name__ for m in matches))
        return matches[0]

    def construct(self, role, window):
        match = self.recognize(role, window)
        return match.provider.construct(match)


class Grammar:
    def accept(self, match):
        return True
