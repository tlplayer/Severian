"""Typed destinations shared by default assignment and grammar expansions."""
from dataclasses import dataclass
from py_compiler.syntax.generic.storage import AddressPlace, BindingPlace, root_binding


def expression_type(node, cfg, env, expected=None):
    """Resolve a capture contract without emitting or executing its expression."""
    node = cfg.capture_expression(node)
    from py_compiler.syntax.block.grammar import expression_grammar
    match = expression_grammar(node, cfg, env)
    if match is not None:
        return match.provider.result
    if node.kind == 'capture':
        return node.operands[0].type(cfg, expected)
    if node.kind in ('name', 'reference'):
        binding = cfg.lookup(node, env)
        if binding is None or binding.type is None:
            raise ValueError('grammar capture requires a resolved binding type')
        return binding.type
    if node.kind == 'literal':
        from py_compiler.frontend.parser.contract import Literal
        from py_compiler.hir.hir.src.program import resolve_literal
        return resolve_literal(Literal('', node.token.span, node.token.text, node.token.kind,
                                       expected.name if expected else None), cfg.syntax, cfg)[0]
    if node.kind == 'unary':
        return expression_type(node.operands[0], cfg, env, expected)
    if node.kind == 'binary':
        left = expression_type(node.operands[0], cfg, env, expected)
        from py_compiler.syntax.generic.owned import select
        grammar = select(left, 'F.operator', node.token.text)
        return cfg.context.type('bool') if getattr(grammar, 'comparison', False) else left
    if node.kind == 'member':
        subject = node.operands[0]
        if subject.kind == 'name' and subject.token.text in cfg.context.package_imports:
            place = cfg.context.imported_globals.get(subject.token.text + '.' + node.token.text)
            if place is None:
                raise ValueError('package has no exported global ' + node.token.text)
            return place.type
        receiver = expression_type(subject, cfg, env)
        for field, name in getattr(getattr(receiver, 'declaration', None), 'fields', ()):
            if field == node.token.text:
                return cfg.context.type(name)
        raise ValueError(f'{receiver.name} has no field {node.token.text!r}')
    if node.kind == 'index':
        receiver = expression_type(node.operands[0], cfg, env)
        if not hasattr(receiver, 'element'):
            raise ValueError(f'{receiver.name} has no indexed storage contract')
        return receiver.element
    if node.kind == 'call':
        callee = node.operands[0]
        if callee.kind == 'name':
            if callee.token.text in cfg.context.types:
                return cfg.context.type(callee.token.text)
            entries = cfg.context.lookup_functions(callee.token.text, cfg.declaration_scope)
        elif callee.kind == 'member':
            subject = callee.operands[0]
            if subject.kind == 'name' and subject.token.text in cfg.context.package_imports:
                name = subject.token.text + '.' + callee.token.text
            else:
                name = expression_type(subject, cfg, env).name + '.' + callee.token.text
            entries = cfg.context.functions.get(name, ())
        else:
            entries = ()
        types = {entry.result for entry in entries if entry.result}
        if len(types) == 1 and all(entry.result for entry in entries):
            return cfg.context.type(types.pop())
    raise ValueError('grammar capture requires an explicit, unambiguous result type')


@dataclass(frozen=True)
class Target:
    term: object
    type: object

    def lower(self, cfg, env, span):
        node = self.term
        if node.kind in ('name', 'reference', 'capture'):
            owner = root_binding(node, cfg, env)
            if owner is None:
                raise ValueError('assignment requires an existing destination')
            place = cfg.storage.get(owner.identity)
            if place is not None:
                receiver = cfg.self_binding if node.kind == 'reference' and node.token.text == 'self' else owner
                return place, receiver, owner.identity not in cfg.uninitialized_fields, owner
            return BindingPlace(owner), owner, True, owner
        if node.kind == 'member':
            subject = node.operands[0]
            if subject.kind == 'name' and subject.token.text in cfg.context.package_imports:
                place = cfg.context.imported_globals[subject.token.text + '.' + node.token.text]
                return place, None, True, None
            owner = root_binding(subject, cfg, env)
            receiver = cfg.expr(subject, env)
            provider = cfg.context.provider_by_name.get(receiver.type.name, receiver.type)
            return provider.field_place(receiver, node.token.text, cfg, span, write=True), owner, True, None
        if node.kind == 'index':
            subject, index = node.operands
            owner = root_binding(subject, cfg, env)
            if owner is None or owner.ownership == 'view':
                raise ValueError('indexed mutation requires an owner or borrow parameter')
            if subject.kind == 'reference' and subject.token.text == 'self':
                owner = cfg.self_binding
            receiver = cfg.expr(subject, env)
            # Protect captured storage while the index expression is evaluated.
            with BindingPlace(owner, rebind=False).capture(cfg, 'borrow'):
                place = receiver.type.element_place(receiver, index, cfg, env, span)
            return place, owner, True, None
        raise ValueError('expression is not an assignable place')


def analyze_target(term, cfg, env):
    term = cfg.capture_expression(term)
    return Target(term, expression_type(term, cfg, env))


def assign(target, rhs, operator, cfg, env, span):
    from py_compiler.syntax.grammar.expression import OWNERSHIP
    from py_compiler.syntax.generic.storage import FieldPlace
    from py_compiler.syntax.generic.owned import select
    if operator == ':=':
        binding = root_binding(target.term, cfg, env)
        if binding is None or binding.identity not in cfg.uninitialized_fields:
            raise ValueError('storage is already declared; use = to update it')
        operator = '='
    mode = rhs.token.text if rhs.kind == 'unary' and rhs.token.text in OWNERSHIP else None
    term = rhs.operands[0] if mode else rhs
    # The default storage grammar chooses RHS-first evaluation.
    value = cfg.expr(term, env, target.type)
    place, owner, initialized, binding = target.lower(cfg, env, span)
    if owner and owner.ownership == 'view':
        raise ValueError('cannot mutate a field through a view; use borrow or copy')
    if getattr(place, 'constant', False):
        raise ValueError('cannot assign to an imported constant')
    if operator == '=' and isinstance(place, FieldPlace):
        from importlib import import_module
        import_module('py_compiler.syntax.block.class').store_field(
            place, rhs, cfg, env, span, owner, initialized, value=value)
        if binding:
            cfg.uninitialized_fields.discard(binding.identity)
        return
    if mode:
        raise ValueError('this destination does not support an ownership modifier')
    address = place if isinstance(place, BindingPlace) else AddressPlace(place, owner)
    if operator == '=':
        with address.capture(cfg, 'borrow'):
            address.write(cfg, value, span)
        from py_compiler.syntax.complex.array import RecordElementPlace
        if isinstance(place, RecordElementPlace) and term.kind == 'call' and callable(getattr(value.type, 'release', None)):
            if not cfg.value_sources.get(value.identity) and value.identity not in cfg.call_result_owners:
                value.type.release(value, cfg, span)
    else:
        select(place.type, 'O.assignment', operator).expand(cfg, address, value, span)
