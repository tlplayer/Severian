from importlib import import_module


class Elif(import_module("py_compiler.syntax.block.if").Conditional):
    spelling = "elif"
    continuation = True
