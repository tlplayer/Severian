"""Collections own their storage operations and default view policy."""


class Collection:
    parameter_ownership = 'view'
    @property
    def binding_ownership(self):
        return 'view'

    def element_place(self, receiver, index, cfg, span):
        raise ValueError(f'{self.name} does not declare an indexed storage grammar')
