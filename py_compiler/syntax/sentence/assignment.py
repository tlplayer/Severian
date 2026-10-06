from dataclasses import replace
from py_compiler.syntax.sentence.syntax import expression, OWNERSHIP
from py_compiler.mir.ownership.flow import Binding


class Assignment:
    def matches(self, node):
        return True

    def lower(self, node, cfg, env, local_names):
        items = list(node.tokens)
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
            if len(items) < 5 or items[2].text not in cfg.syntax.types:
                raise ValueError("binding requires a resolved primitive annotation and initializer")
            annotation, position = cfg.syntax.types[items[2].text], 3
        if position >= len(items) or items[position].text not in ("=", ":=", "+=", "-=", "*="):
            raise ValueError("expected a binding or assignment")
        operator = items[position].text
        rhs = expression(items[position + 1:])
        mode = rhs.token.text if rhs.kind == "unary" and rhs.token.text in OWNERSHIP else None
        core = rhs.operands[0] if mode else rhs
        owner = cfg.lookup(core, env) if core.kind in ("name", "reference") else None
        existing = destination.get(first.text)
        if not qualified and existing and first.text not in local_names and (operator == ":=" or annotation):
            existing = None
        if operator == ":=" and existing:
            raise ValueError(f"binding {first.text!r} already exists in this scope")
        if operator in ("+=", "-=", "*="):
            if not existing:
                raise ValueError("compound assignment requires an existing binding")
            cfg.flow.read(existing)
        expected = annotation or (existing.type if existing else None)
        value = cfg.expr(core, env, expected)
        if expected and value.type != expected:
            raise ValueError(f"initializer {value.type.name} does not satisfy {expected.name}")
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
        if existing:
            previous_value = cfg.read_binding(existing, node.span) if operator in ("+=", "-=", "*=") else None
            cfg.flow.replace(existing)
            if existing.type != value.type:
                raise ValueError("assignment changes the binding's type")
            if operator in ("+=", "-=", "*="):
                if value.type.family not in ("integer", "float", "byte"):
                    raise ValueError("compound assignment requires numeric operands")
                value = cfg.emit("binary", value.type, (previous_value, value), operator[0], node.span)
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
        cfg.values[binding.identity] = value
        if binding.identity in cfg.storage:
            cfg.storage[binding.identity].write(cfg, value, node.span)
        if owner and ownership in ("view", "borrow"):
            cfg.flow.loan(binding, owner)
