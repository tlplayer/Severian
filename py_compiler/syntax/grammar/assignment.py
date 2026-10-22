from dataclasses import replace
from py_compiler.syntax.grammar.expression import expression, OWNERSHIP
from py_compiler.mir.ownership.flow import Binding
from py_compiler.syntax.generic.owned import select
from py_compiler.syntax.generic.storage import BindingPlace, AddressPlace, root_binding


class Assignment:
    def matches(self, node):
        return True

    def lower(self, node, cfg, env, local_names, *, declare_local=False, rhs_override=None):
        items = list(node.tokens)
        from py_compiler.syntax.catalog import assignment_spellings
        assignments = assignment_spellings(cfg.syntax)
        # Object stores go through the class storage provider.
        depth, assignment = 0, None
        for i, token in enumerate(items):
            if token.text in ('[', '('):
                depth += 1
            elif token.text in (']', ')'):
                depth -= 1
            elif depth == 0 and token.text in assignments:
                assignment = i
                break
        if assignment is not None:
            left_tokens = items[:assignment]
            if not left_tokens:
                raise ValueError('assignment requires a target')
            pattern = left_tokens[1:-1] if left_tokens[0].text == '(' and left_tokens[-1].text == ')' else left_tokens
            depth, unpacking = 0, False
            for token in pattern:
                if token.text in ('(', '['):
                    depth += 1
                elif token.text in (')', ']'):
                    depth -= 1
                elif token.text == ',' and depth == 0:
                    unpacking = True
            if unpacking:
                self.unpack(node, pattern, items[assignment], items[assignment + 1:], cfg, env, local_names, declare_local)
                return
            # An annotation introduces a binding; other targets compose as expressions.
            prefix = 2 if (len(left_tokens) > 2 and left_tokens[0].text in cfg.syntax.scope_providers
                           and left_tokens[1].text == '.') else 0
            annotation = len(left_tokens) > prefix + 1 and left_tokens[prefix + 1].text == ':'
            if not annotation:
                left = cfg.capture_expression(expression(left_tokens))
                if left.kind in ('member', 'index', 'capture') or (left.kind == 'reference' and left.token.text == 'self'):
                    from py_compiler.syntax.generic.place import analyze_target, assign
                    assign(analyze_target(left, cfg, env), cfg.capture_expression(expression(items[assignment + 1:])),
                           items[assignment].text, cfg, env, node.span)
                    return
        destination = env
        qualified = None
        if len(items) > 2 and items[0].text in cfg.syntax.scope_providers and items[1].text == ".":
            qualified = items[0].text
            destination = cfg.syntax.scope_providers[qualified].bindings(cfg, env)
            items = items[2:]
        first = items[0]
        if first.kind != "IDENTIFIER":
            raise ValueError(f"keyword {first.text!r} has no executable sentence provider")
        position, annotation = 1, None
        if len(items) > 1 and items[1].text == ":":
            if len(items) > 3 and items[2].text in cfg.syntax.scope_providers:
                if qualified and qualified != items[2].text:
                    raise ValueError("binding scope qualifier and annotation scope disagree")
                qualified = items[2].text
                destination = cfg.syntax.scope_providers[qualified].bindings(cfg, env)
                items = items[:2] + items[3:]
            position = next((i for i in range(2, len(items)) if items[i].text in assignments), len(items))
            spelling = ''.join(t.text for t in items[2:position])
            annotation = None if spelling == 'infer' else cfg.context.type(spelling)
        if position >= len(items) or items[position].text not in assignments:
            raise ValueError("expected a binding or assignment")
        operator = items[position].text
        capture_error = operator == '?='
        if capture_error:
            operator = '='
        rhs = rhs_override or expression(items[position + 1:])
        mode = rhs.token.text if rhs.kind == "unary" and rhs.token.text in OWNERSHIP else None
        core = rhs.operands[0] if mode else rhs
        atomic_share = mode == 'share' and core.kind == 'atomic-share'
        if atomic_share:
            core = core.operands[0]
        owner = cfg.lookup(core, env) if core.kind in ("name", "reference") else None
        if core.kind == 'call' and core.operands[0].kind == 'index' and core.operands[0].operands[0].token.text == 'pointer':
            source = core.operands[1] if len(core.operands) == 2 else None
            owner = root_binding(source, cfg, env) if source else None
            if owner is None:
                raise ValueError('pointer conversion requires a live array binding')
        if core.kind == 'address':
            owner = root_binding(core.operands[0], cfg, env)
            if mode in ('move', 'copy', 'mirror'):
                raise ValueError('address of local storage cannot transfer allocation ownership')
        existing = None if declare_local else destination.get(first.text)
        if not declare_local and not existing and qualified is None and first.text in cfg.namespaces.get("self", {}):
            destination = cfg.namespaces["self"]
            existing = destination[first.text]
            qualified = "self"
        if (not declare_local and not existing and qualified is None and annotation is None
                and operator != ':=' and first.text in cfg.namespaces.get('module', {})):
            destination = cfg.namespaces['module']
            existing = destination[first.text]
            qualified = 'module'
        if existing and qualified == 'self' and existing.ownership == 'view':
            raise ValueError('cannot mutate a field through a view receiver')
        if not qualified and existing and first.text not in local_names and (operator == ":=" or annotation):
            existing = None
        if existing and qualified == "self" and existing.identity in cfg.uninitialized_fields and operator == ":=":
            operator = "="
        if operator == ":=" and existing:
            raise ValueError(f"binding {first.text!r} already exists in this scope")
        if operator not in ("=", ":="):
            if not existing:
                raise ValueError("compound assignment requires an existing binding")
            cfg.flow.read(existing)
        if existing and qualified == 'self' and operator == '=':
            from importlib import import_module
            store_field = import_module('py_compiler.syntax.block.class').store_field
            store_field(cfg.storage[existing.identity], rhs, cfg, env, node.span,
                        cfg.self_binding, existing.identity not in cfg.uninitialized_fields)
            cfg.uninitialized_fields.discard(existing.identity)
            return
        if existing and existing.type.family == 'unit' and operator not in ('=', ':='):
            if mode:
                raise ValueError('unit compound assignment does not take an ownership modifier')
            value = cfg.expr(core, env)
            select(existing.type, 'O.assignment', operator).expand(cfg, BindingPlace(existing), value, node.span)
            return
        expected = annotation or (existing.type if existing else None)
        if expected is None and core.kind == 'call' and len(core.operands) == 1 and core.operands[0].kind == 'name':
            from py_compiler.syntax.complex.generic import infer_empty_constructor
            expected = infer_empty_constructor(core.operands[0].token.text, first.text,
                                              getattr(cfg, 'remaining_nodes', ()), cfg, env)
        if expected and expected.family == 'shared' and owner and owner.type != expected and mode != 'share':
            raise ValueError('use share to convert a named owner into shared ownership')
        moving_place = None
        if mode == 'move' and core.kind in ('member', 'index'):
            from py_compiler.syntax.generic.place import analyze_target
            moving_place, moving_owner, _, _ = analyze_target(core, cfg, env).lower(cfg, env, node.span)
            reader = getattr(moving_place, 'read_view', moving_place.read)
            value = reader(cfg, node.span)
        elif mode in ('view', 'borrow') and core.kind == 'index':
            subject, index = core.operands
            owner = root_binding(subject, cfg, env)
            receiver = cfg.expr(subject, env)
            place = receiver.type.element_place(receiver, index, cfg, env, node.span)
            value = place.read_view(cfg, node.span, mode) if hasattr(place, 'read_view') else place.read(cfg, node.span)
        else:
            previous_capture = getattr(cfg, 'captured_error_expression', None)
            if capture_error:
                cfg.captured_error_expression = core
            try:
                value = cfg.expr(core, env, None if mode == 'share' else expected)
            finally:
                cfg.captured_error_expression = previous_capture
        if mode == 'share':
            from py_compiler.syntax.complex.shared import Shared
            if owner is None or owner.ownership in ('view', 'borrow'):
                raise ValueError('share requires an owning binding')
            cfg.flow.transfer(owner)
            if not isinstance(value.type, Shared):
                if any(source == owner.identity for source in cfg.flow.views.values()):
                    raise ValueError('cannot share while views or borrows are live')
                target = Shared(value.type, cfg.context.type('usize').bits, atomic_share)
                value = target.from_value(value, cfg, node.span)
                updated_owner = replace(owner, type=target, ownership='own')
                env[owner.name] = updated_owner
                cfg.values[owner.identity] = value
                cfg.body.bindings = [updated_owner if b.identity == owner.identity else b for b in cfg.body.bindings]
                owner = updated_owner
            elif value.type.atomic != atomic_share:
                raise ValueError('shared reference atomicity cannot change through sharing')
            value = value.type.copy_value(value, cfg, node.span)
            mode, owner = 'own', None
        if core.kind in ('member', 'index') and value is not None and value.identity not in getattr(cfg, 'owned_results', ()) and value.type.family in ('record', 'array', 'tensor', 'tuple', 'pointer', 'shared', 'string', 'union'):
            owner = root_binding(core, cfg, env)
        if owner and owner.identity in cfg.storage:
            from py_compiler.syntax.generic.storage import FieldPlace
            if isinstance(cfg.storage[owner.identity], FieldPlace):
                if mode == 'move':
                    raise ValueError('moving an owned field requires replacing it; use copy')
                owner = cfg.self_binding
        if value is None:
            raise ValueError("a no-result call cannot initialize a binding")
        creating_mirror = mode == 'mirror'
        if creating_mirror:
            from py_compiler.syntax.complex.mirror import capture
            value = capture(value, owner, cfg, env, node.span)
            mode, owner = 'own', None
        if value.identity in cfg.owned_results and (owner is None or owner.type != value.type):
            owner = None
        if value.identity in cfg.call_result_owners:
            owner = cfg.call_result_owners[value.identity]
            mode = mode or cfg.call_result_modes.get(value.identity)
        if expected and value.type != expected:
            raise ValueError(f"initializer {value.type.name} does not satisfy {expected.name}")
        if value is None:
            raise ValueError("a no-result call cannot initialize a binding")
        if owner and callable(getattr(owner.type, 'release', None)) and (core.kind == 'address' or value.type.family == 'pointer') and mode in ('move', 'copy', 'mirror'):
            raise ValueError('address conversion cannot transfer array allocation ownership; use view or borrow')
        if mode in ("copy", "mirror") and callable(getattr(value.type, 'copy_value', None)):
            value = value.type.copy_value(value, cfg, node.span)
        elif mode == "copy":
            provider = cfg.context.provider_by_name.get(value.type.name) if cfg.context else None
            if hasattr(provider, "copy_value"):
                value = provider.copy_value(value, cfg, node.span)
        if owner is None and mode == 'view' and value.type.binding_ownership == 'copy':
            mode = 'copy'
        ownership = mode or ("copy" if value.type.binding_ownership == "copy" else ("view" if owner else "own"))
        # String bytes are immutable here: copy captures a value, never a binding.
        # A future mutable string provider must supply allocation/COW behavior.
        if mode in ("borrow", "view", "move") and owner is None:
            raise ValueError(f"{mode} requires an existing binding")
        if mode == "move":
            from py_compiler.syntax.generic.storage import owns_storage
            if core.kind in ('member', 'index') and owns_storage(value.type):
                place, source = moving_place, moving_owner
                if not callable(getattr(place, 'take', None)):
                    raise ValueError('this projection does not support ownership transfer')
                if source is None or source.ownership in ('view', 'borrow'):
                    raise ValueError('field move requires an owning binding')
                with AddressPlace(place, source).capture(cfg, 'borrow'):
                    value = place.take(cfg, node.span, value)
                ownership, mode, owner = 'own', 'own', None
            else:
                if owner and owner.identity in cfg.storage and owns_storage(value.type):
                    raise ValueError('moving an owned field requires an explicit field projection')
                viewed = cfg.flow.views.get(owner.identity)
                cfg.flow.move(owner)
                ownership = owner.ownership if viewed else "own"
                if viewed:
                    owner = next((b for b in env.values() if b.identity == viewed), None)
                    if owner is None:
                        raise ValueError("moved view has no live source binding")
        if existing and operator not in ('=', ':='):
            select(existing.type, 'O.assignment', operator).expand(cfg, BindingPlace(existing), value, node.span)
            return
        if existing:
            if callable(getattr(existing.type, 'check_assignment', None)):
                existing.type.check_assignment(cfg.read_binding(existing, node.span), value, cfg)
            release_string = existing.type.family == 'string' and existing.ownership == 'own' and existing.identity not in cfg.flow.moved and ownership == 'own'
            release_union = existing.type.family == 'union' and existing.ownership == 'own' and existing.identity not in cfg.flow.moved and ownership == 'own'
            if callable(getattr(existing.type, 'release', None)) and existing.identity not in cfg.flow.moved and not (release_string or release_union):
                raise ValueError('memory rebinding requires moving or releasing the previous owner')
            cfg.flow.replace(existing)
            if release_string or release_union:
                previous = cfg.read_binding(existing, node.span)
                from py_compiler.syntax.generic.storage import release_storage
                release_storage(previous, cfg, node.span)
            if existing.type != value.type:
                raise ValueError("assignment changes the binding's type")
            binding = replace(existing, ownership=ownership)
            destination[first.text] = binding
            if qualified is None or qualified == "local":
                env[first.text] = binding
            cfg.flow.views.pop(binding.identity, None)
            cfg.flow.end_loan(binding.identity)
        else:
            cfg.context.check_prelude_conflict(first.text)
            binding = Binding(node.identity, first.text, value.type, operator == ":=", ownership, node.span)
            destination[first.text] = binding
            if qualified is None or qualified == "local":
                env[first.text] = binding
                local_names.add(first.text)
            cfg.body.bindings.append(binding)
            if cfg.storage_factory and not declare_local and ownership not in ("view", "borrow"):
                place = cfg.storage_factory(binding)
                place.declaration()
                cfg.storage[binding.identity] = place
                cfg.body.storage.append(place)
        cfg.uninitialized_fields.discard(binding.identity)
        cfg.record_lifetime('initialize' if existing else 'declare', binding.identity, binding.name)
        cfg.values[binding.identity] = value
        if creating_mirror:
            cfg.flow.mirror_readers.add(binding.identity)
        dependencies = cfg.value_dependencies.get(value.identity, set())
        if ownership not in ('view', 'borrow'):
            if owner and ownership in ('copy', 'mirror', 'move'):
                cfg.flow.copy_retained(binding.identity, owner.identity)
            else:
                cfg.flow.retain(binding.identity, '*', dependencies)
            if mode == 'move' and owner:
                cfg.flow.end_retained(owner.identity)
        if binding.identity in cfg.storage:
            cfg.storage[binding.identity].write(cfg, value, node.span)
        if owner and ownership in ("view", "borrow"):
            sources = cfg.call_result_sources.get(value.identity)
            if sources:
                cfg.flow.view_sources[binding.identity] = sources
                cfg.flow.read(binding)
                if ownership == 'borrow':
                    cfg.flow.retain(binding.identity, '*', sources)
            else:
                cfg.flow.loan(binding, owner)

    def unpack(self, node, pattern, operator, rhs, cfg, env, local_names, declare_local):
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        from py_compiler.syntax.complex.tuple import ProductType, TupleValue
        from py_compiler.syntax.generic.storage import copy_storage, release_storage
        if operator.text not in ('=', ':='):
            raise ValueError('tuple unpacking requires = or :=')
        from py_compiler.syntax.block.test import groups
        targets = [expression(tokens) for tokens in groups(pattern)]
        if any(target.kind not in ('name', 'index', 'member', 'reference') for target in targets):
            raise ValueError('tuple unpacking requires assignable targets')
        visible = [target.token.text for target in targets if target.kind == 'name' and target.token.text != '_']
        if len(set(visible)) != len(visible):
            raise ValueError('tuple unpacking requires distinct binding names')
        parts = groups(rhs)
        term = Expression('tuple', rhs[0], tuple(expression(part) for part in parts)) if len(parts) > 1 else expression(rhs)
        value = cfg.expr(term, env)
        count = (len(value.type.generic_arguments) if isinstance(value.type, ProductType)
                 else value.type.count if isinstance(value.type, TupleValue) else None)
        if count != len(targets):
            raise ValueError('tuple pattern does not match tuple arity')
        # Snapshot every member before assigning any target (a, b = b, a).
        # Owned members are copied, so unpacked values outlive the tuple.
        fields = []
        for index, target in enumerate(targets):
            token = target.token
            if target.kind == 'name' and token.text == '_':
                continue
            position = Expression('literal', token.with_text(str(index), kind='NUMBER'))
            field = value.type.element_place(value, position, cfg, env, node.span).read(cfg, node.span)
            fields.append((target, copy_storage(field, cfg, node.span)))
        if value.identity in cfg.owned_results and root_binding(term, cfg, env) is None:
            release_storage(value, cfg, node.span)
        for index, (target, field) in enumerate(fields):
            token = target.token
            capture = Expression('capture', token, (ArgumentValue(field),))
            if target.kind != 'name':
                from py_compiler.syntax.generic.place import analyze_target, assign
                assign(analyze_target(target, cfg, env), capture, operator.text, cfg, env, node.span)
                continue
            member = replace(node, identity=node.identity + ':unpack:' + str(index), tokens=(token, operator, token))
            self.lower(member, cfg, env, local_names, declare_local=declare_local, rhs_override=capture)


import unittest


class TupleAssignmentTests(unittest.TestCase):
    def compile(self, statements):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'test "unpack":\n' + ''.join('    ' + line + '\n' for line in statements.splitlines())
        return compile_source('tuple-assignment.sev', source, Syntax(include_tests=True))

    def test_indexed_swap_snapshots_both_values(self):
        result = self.compile('values = tensor[int](2)\nvalues[0] = 3\nvalues[1] = 7\nvalues[0], values[1] = values[1], values[0]\nassert(values[0] == 7 and values[1] == 3)')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_unpack_and_swap_evaluate_values_before_assignment(self):
        result = self.compile('left, right = (3, 7)\nleft, right = (right, left)\nassert(left == 7 and right == 3)')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_discard_and_parenthesized_pattern(self):
        result = self.compile('(left, _, right) = (3, 5, 7)\nassert(left == 3 and right == 7)')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_invalid_patterns_and_reassignment_are_rejected(self):
        cases = (
            'left, right = (1, 2, 3)',
            'left, left = (1, 2)',
            'left, right = 3',
            'left, right := (1, 2)\nleft, right = (3, 4)',
            'left, right = (1, 2)\nleft, right += (3, 4)',
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertTrue(self.compile(source).diagnostics)
