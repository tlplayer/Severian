"""Typed destinations shared by default assignment and grammar expansions."""
from dataclasses import dataclass
from py_compiler.syntax.generic.storage import AddressPlace, BindingPlace, root_binding


def condition_type(node, cfg, env):
    if node.kind in ('list', 'map', 'set') and not node.operands:
        return cfg.context.type('bool')
    actual = expression_type(node, cfg, env)
    if actual.family != 'bool':
        from py_compiler.syntax.block.grammar import boolean_provider
        boolean_provider(actual, cfg)
    return cfg.context.type('bool')


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
    if node.kind in ('list', 'map', 'set'):
        if expected is not None:
            return expected
        width = 2 if node.kind == 'map' else 1
        if len(node.operands) < width:
            raise ValueError('empty collection literal requires a type annotation')
        types = tuple(expression_type(t, cfg, env).name for t in node.operands[:width])
        template = cfg.context.record_templates[cfg.context.declaration_name(node.kind, cfg.declaration_scope)]
        return template.realize(types, cfg.context)
    if node.kind == 'unary':
        if node.token.text in ('not', '!'):
            condition_type(node.operands[0], cfg, env)
            return cfg.context.type('bool')
        return expression_type(node.operands[0], cfg, env, expected)
    if node.kind == 'binary':
        if node.token.text in ('and', 'or'):
            for operand in node.operands:
                condition_type(operand, cfg, env)
            return cfg.context.type('bool')
        left = expression_type(node.operands[0], cfg, env, expected)
        right = expression_type(node.operands[1], cfg, env)
        if left.family == 'unit' or right.family == 'unit':
            from py_compiler.syntax.complex.units import result_type
            return result_type(node.token.text, left, right, cfg.context)
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
        if receiver.family == 'shared':
            receiver = receiver.payload
        if receiver.family == 'union' and node.token.text in ('has_value', 'value'):
            members = [member for member in receiver.variants if member.mlir]
            if members and len(members) < len(receiver.variants):
                from py_compiler.syntax.block.union import Union
                return cfg.context.type('bool') if node.token.text == 'has_value' else Union().resolve(members, cfg.context)
        for field, name in getattr(getattr(receiver, 'declaration', None), 'fields', ()):
            if field == node.token.text:
                return cfg.context.type(name)
        raise ValueError(f'{receiver.name} has no field {node.token.text!r}')
    if node.kind == 'index':
        receiver = expression_type(node.operands[0], cfg, env)
        if node.operands[1].kind == 'slice':
            if receiver.family == 'string':
                return receiver
            if receiver.family in ('tensor', 'tuple') and hasattr(receiver, 'element'):
                from py_compiler.syntax.complex.tensor import TensorType
                return TensorType(receiver.element, (None,))
            if receiver.family == 'record':
                entries = cfg.context.functions.get(receiver.name + '.__slice', ())
                results = {entry.result for entry in entries}
                if len(results) == 1 and None not in results:
                    result = cfg.context.type(results.pop())
                    from py_compiler.syntax.complex.error import is_error
                    if result.family == 'union' and not getattr(cfg, 'preserve_error_result', False):
                        successes = [member for member in result.variants if not is_error(member)]
                        if len(successes) == 1:
                            return successes[0]
                    return result
            raise ValueError(f'{receiver.name} has no slicing result contract')
        if callable(getattr(receiver, 'indexed_type', None)):
            return receiver.indexed_type(node.operands[1])
        if receiver.family == 'record':
            entries=cfg.context.functions.get(receiver.name+'.get',())
            results={e.result for e in entries}
            if len(results)==1:
                return cfg.context.type(results.pop())
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
            from py_compiler.syntax.complex.collection import indexed_write_preserves_shape
            stable = indexed_write_preserves_shape(receiver.type, cfg.context)
            with BindingPlace(owner, rebind=False).capture(cfg, 'borrow', preserve_shape=stable):
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
    value = cfg.expr(term, env, None if target.type.family == 'unit' and operator != '=' else target.type)
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
    if mode and (operator != '=' or mode not in ('move', 'copy')):
        raise ValueError('this destination does not support an ownership modifier')
    if initialized and callable(getattr(place.type, 'check_assignment', None)):
        place.type.check_assignment(place.read(cfg, span), value, cfg)
    address = place if isinstance(place, BindingPlace) else AddressPlace(place, owner)
    if operator == '=':
        source = root_binding(term, cfg, env)
        if mode == 'move':
            from py_compiler.syntax.generic.storage import owns_storage
            if source and (source.ownership in ('view', 'borrow') or source.identity in cfg.storage or term.kind not in ('name', 'reference', 'capture')):
                raise ValueError('element ownership transfer requires a whole owned binding')
            if source and owner and cfg.flow.resolve(source.identity) == cfg.flow.resolve(owner.identity):
                raise ValueError('element cannot consume its containing object')
            if source is None and owns_storage(value.type) and value.identity not in cfg.owned_results:
                raise ValueError('element cannot retain a borrowed result; use copy')
            if owns_storage(value.type) and not callable(getattr(place, 'write_owned', None)):
                raise ValueError('destination does not support owned element storage')
        from py_compiler.syntax.generic.storage import contains_borrowed_fields, element_slot
        if owner and target.term.kind == 'index' and contains_borrowed_fields(value.type, cfg.context):
            identity = cfg.flow.resolve(owner.identity)
            slot = element_slot(target.term)
            dependencies = set(cfg.value_dependencies.get(value.identity, ()))
            if slot == '*':
                dependencies.update(cfg.flow.sources(identity))
            cfg.flow.retain(identity, slot, dependencies)
        with address.capture(cfg, 'borrow'):
            if mode == 'move':
                if source:
                    cfg.flow.move(source)
                    cfg.flow.end_retained(source.identity)
                getattr(place, 'write_owned', place.write)(cfg, value, span)
            else:
                address.write(cfg, value, span)
        from py_compiler.syntax.complex.array import RecordElementPlace
        if mode != 'move' and isinstance(place, RecordElementPlace) and term.kind == 'call' and callable(getattr(value.type, 'release', None)):
            if not cfg.value_sources.get(value.identity) and value.identity not in cfg.call_result_owners:
                value.type.release(value, cfg, span)
    else:
        select(place.type, 'O.assignment', operator).expand(cfg, address, value, span)
