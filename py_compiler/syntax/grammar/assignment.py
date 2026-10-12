from dataclasses import replace
from py_compiler.syntax.grammar.expression import expression, OWNERSHIP
from py_compiler.mir.ownership.flow import Binding
from py_compiler.syntax.generic.owned import select
from py_compiler.syntax.generic.storage import BindingPlace, AddressPlace, root_binding


class Assignment:
    def matches(self, node):
        return True

    def lower(self, node, cfg, env, local_names, *, declare_local=False):
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
            # An annotation introduces a binding; other targets compose as expressions.
            annotation = len(left_tokens) > 1 and left_tokens[1].text == ':'
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
            annotation = cfg.context.type(''.join(t.text for t in items[2:position]))
        if position >= len(items) or items[position].text not in assignments:
            raise ValueError("expected a binding or assignment")
        operator = items[position].text
        rhs = expression(items[position + 1:])
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
            owner = cfg.lookup(core.operands[0], env)
            if mode in ('move', 'copy', 'mirror'):
                raise ValueError('address of local storage cannot transfer allocation ownership')
        existing = None if declare_local else destination.get(first.text)
        if not declare_local and not existing and qualified is None and first.text in cfg.namespaces.get("self", {}):
            destination = cfg.namespaces["self"]
            existing = destination[first.text]
            qualified = "self"
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
        expected = annotation or (existing.type if existing else None)
        if expected and expected.family == 'shared' and owner and owner.type != expected and mode != 'share':
            raise ValueError('use share to convert a named owner into shared ownership')
        value = cfg.expr(core, env, None if mode == 'share' else expected)
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
        if core.kind in ('member', 'index') and value is not None and value.identity not in getattr(cfg, 'owned_results', ()) and value.type.family in ('record', 'array', 'pointer', 'shared'):
            owner = root_binding(core, cfg, env)
        if owner and owner.identity in cfg.storage:
            from py_compiler.syntax.generic.storage import FieldPlace
            if isinstance(cfg.storage[owner.identity], FieldPlace):
                if mode == 'move':
                    raise ValueError('moving an owned field requires replacing it; use copy')
                owner = cfg.self_binding
        if value is None:
            raise ValueError("a no-result call cannot initialize a binding")
        if owner and value.type.family == 'dynamic' and owner.type != value.type:
            owner = None
        if core.kind == 'call' and value.identity in cfg.call_result_owners:
            owner = cfg.call_result_owners[value.identity]
        if expected and value.type != expected:
            raise ValueError(f"initializer {value.type.name} does not satisfy {expected.name}")
        if value is None:
            raise ValueError("a no-result call cannot initialize a binding")
        if owner and callable(getattr(owner.type, 'release', None)) and not callable(getattr(value.type, 'release', None)) and mode in ('move', 'copy', 'mirror'):
            raise ValueError('address conversion cannot transfer array allocation ownership; use view or borrow')
        if mode in ("copy", "mirror") and callable(getattr(value.type, 'copy_value', None)):
            value = value.type.copy_value(value, cfg, node.span)
        elif mode == "copy":
            provider = cfg.context.provider_by_name.get(value.type.name) if cfg.context else None
            if hasattr(provider, "copy_value"):
                value = provider.copy_value(value, cfg, node.span)
        ownership = mode or ("copy" if value.type.binding_ownership == "copy" else ("view" if owner else "own"))
        # String bytes are immutable here: copy captures a value, never a binding.
        # A future mutable string provider must supply allocation/COW behavior.
        if mode in ("borrow", "view", "move") and owner is None:
            raise ValueError(f"{mode} requires an existing binding")
        if mode == "move":
            if owner and owner.identity in cfg.storage and callable(getattr(value.type, 'release', None)):
                raise ValueError('moving an owned field requires replacing it; use copy')
            if core.kind in ('member', 'index') and callable(getattr(value.type, 'release', None)):
                raise ValueError('moving an owned field requires replacing it; use copy')
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
            if callable(getattr(existing.type, 'release', None)) and existing.identity not in cfg.flow.moved:
                raise ValueError('memory rebinding requires moving or releasing the previous owner')
            cfg.flow.replace(existing)
            if existing.type != value.type:
                raise ValueError("assignment changes the binding's type")
            binding = replace(existing, ownership=ownership)
            destination[first.text] = binding
            if qualified is None or qualified == "local":
                env[first.text] = binding
            cfg.flow.views.pop(binding.identity, None)
            cfg.flow.loans.pop(binding.identity, None)
        else:
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
        cfg.values[binding.identity] = value
        if binding.identity in cfg.storage:
            cfg.storage[binding.identity].write(cfg, value, node.span)
        if owner and ownership in ("view", "borrow"):
            cfg.flow.loan(binding, owner)
