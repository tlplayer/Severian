from dataclasses import dataclass


@dataclass(frozen=True)
class Primitive:
    name: str
    family: str
    mlir: str
    bits: int = 0
    signed: bool = False

    @property
    def binding_ownership(self):
        return "view" if self.family == "string" else "copy"
