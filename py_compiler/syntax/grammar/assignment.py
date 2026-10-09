from dataclasses import replace
from py_compiler.syntax.grammar.expression import expression, OWNERSHIP
from py_compiler.mir.ownership.flow import Binding
from py_compiler.syntax.generic.owned import select
from py_compiler.syntax.generic.storage import BindingPlace, AddressPlace, root_binding


class Assignment:
    def matches(self, node):
        return True

    def lower(self, node, cfg, env, local_names):
        items = list(node.tokens)
        from py_compiler.syntax.catalog import assignment_spellings
        assignments = assignment_spellings(cfg.syntax)
        # Object stores go through the class storage provider.
        assignment = next((i for i, t in enumerate(items) if t.text in assignments), None)
        if assignment is not None and any(t.text == '.' for t in items[:assignment]) and items[0].text not in cfg.syntax.scope_providers:
            left = expression(items[:assignment])
            if left.kind != 'member':
                raise ValueError('assignment target is not a field')
            subject = left.operands[0]
            owner = root_binding(subject, cfg, env)
            if owner and owner.ownership == 'view':
                raise ValueError('cannot mutate a field through a view; use borrow or copy')
            receiver = cfg.expr(subject, env)
            provider = cfg.context.provider_by_name.get(receiver.type.name)
            if not hasattr(provider, 'field_place'):
                raise ValueError('receiver has no field storage provider')
            place = provider.field_place(receiver, left.token.text, cfg, node.span, write=True)
            operator = items[assignment].text
            if operator == ':=':
                raise ValueError('a field is already declared; use = to update it')
            value = cfg.expr(expression(items[assignment + 1:]), env, place.type)
            if operator != '=':
                select(place.type, 'O.assignment', operator).expand(cfg, AddressPlace(place, owner), value, node.span)
            else:
                place.write(cfg, value, node.span)
            return
        if assignment is not None and any(t.text == '[' for t in items[:assignment]) and not any(t.text == ':' for t in items[:assignment]):
            left = expression(items[:assignment])
            if left.kind != 'index':
                raise ValueError('assignment target is not an indexed place')
            subject, index = left.operands
            owner = cfg.lookup(subject, env) if subject.kind in ('name', 'reference') else None
            if owner is None or owner.ownership == 'view':
                raise ValueError('indexed mutation requires an owner or borrow parameter')
            receiver = cfg.expr(subject, env)
            place = receiver.type.element_place(receiver, index, cfg, env, node.span)
            value = cfg.expr(expression(items[assignment + 1:]), env, place.type)
            operator = items[assignment].text
            if operator == '=':
                place.write(cfg, value, node.span)
            elif operator == ':=':
                raise ValueError('indexed storage is already declared')
            else:
                select(place.type, 'O.assignment', operator).expand(cfg, AddressPlace(place, owner), value, node.span)
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
        owner = cfg.lookup(core, env) if core.kind in ("name", "reference") else None
        if core.kind == 'call' and core.operands[0].kind == 'index' and core.operands[0].operands[0].token.text == 'pointer':
            source = core.operands[1] if len(core.operands) == 2 else None
            owner = cfg.lookup(source, env) if source and source.kind in ('name', 'reference') else None
            if owner is None:
                raise ValueError('pointer conversion requires a live array binding')
        existing = destination.get(first.text)
        if not existing and qualified is None and first.text in cfg.namespaces.get("self", {}):
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
        expected = annotation or (existing.type if existing else None)
        value = cfg.expr(core, env, expected)
        if core.kind == 'member' and value is not None and value.type.family == 'record':
            owner = root_binding(core, cfg, env)
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
            if callable(getattr(existing.type, 'release', None)):
                raise ValueError('memory rebinding requires a new binding; release the previous owner explicitly')
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
            if cfg.storage_factory and ownership not in ("view", "borrow"):
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
