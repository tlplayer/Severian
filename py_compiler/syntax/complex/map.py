"""Brace literals and map[K,V,C] default to ordered B-tree storage."""
from dataclasses import dataclass
from py_compiler.syntax.complex.dictionary import Dictionary


@dataclass(frozen=True)
class Map(Dictionary):
    name: str = 'map'
    family: str = 'map'
