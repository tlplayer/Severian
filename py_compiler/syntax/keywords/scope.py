"""Compatibility catalog; block classes own scope qualifier behavior."""
from importlib import import_module
from py_compiler.syntax.block.function import Function as Local


SCOPES = {'local': Local(), 'module': import_module('py_compiler.syntax.block.global').Global(),
          'self': import_module('py_compiler.syntax.block.class').Class()}
