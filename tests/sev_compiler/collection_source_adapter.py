"""Algorithm-only execution of collection source bodies, NOT a Sev compiler.

This deliberately narrow adapter erases annotations and maps primitive storage
and method overloads to Python. Algorithms are read from the .sev files on each
run, never reimplemented here. It cannot validate layouts, traits, loans,
move/drop, allocation failures, or native integer semantics. Native acceptance
is measured independently by collection_capabilities.py.
"""
import builtins
import copy
from pathlib import Path
import re

DIRECTORY = Path(__file__).resolve().parents[2] / 'sev_compiler/universal/collections'


class Scalar(int):
    max = (1 << 64) - 1

    def __new__(cls, value=0, mode=None):
        if mode == 'lossy':
            value = int(value) & cls.max
        return super().__new__(cls, value)

    @staticmethod
    def default():
        return 0


class Key(int):
    def hash(self):
        return int(self) & Scalar.max


class Slots:
    """Copy-return initialized slots, with capacity independent of length."""
    def __init__(self, values=()):
        self.data = builtins.list(values)
        self.capacity = len(self.data)

    def __class_getitem__(cls, item):
        return cls

    def reserve(self, n):
        assert n >= 0
        self.capacity = max(self.capacity, len(self.data) + n)

    def append(self, value):
        self.data.append(copy.copy(value))
        self.capacity = max(self.capacity, len(self.data))

    def pop(self, index=None):
        if not self.data or (index is not None and index >= len(self.data)):
            return (0, False)
        return (self.data.pop(-1 if index is None else index), True)

    def __getitem__(self, index):
        assert 0 <= index < len(self.data)
        return copy.copy(self.data[index])

    def __setitem__(self, index, value):
        assert 0 <= index < len(self.data)
        self.data[index] = copy.copy(value)

    def __iter__(self):
        return iter(self.data)

    def len(self):
        return len(self.data)

    def cap(self):
        return self.capacity

    def clear(self):
        self.data.clear()


class Memory:
    def __init__(self, n):
        self.data = [None] * n
        self.freed = False

    def __getitem__(self, index):
        assert not self.freed and 0 <= index < len(self.data)
        assert self.data[index] is not None, 'uninitialized read'
        return copy.copy(self.data[index])

    def __setitem__(self, index, value):
        assert not self.freed and 0 <= index < len(self.data)
        self.data[index] = copy.copy(value)


def free(memory):
    assert not memory.freed, 'double free'
    memory.freed = True


class Slice:
    def __init__(self, memory, start, length):
        self.memory, self.start, self.length = memory, start, length

    def len(self):
        return self.length

    def __getitem__(self, index):
        assert 0 <= index < self.length
        return self.memory[self.start + index]

    def __iter__(self):
        for index in range(self.length):
            yield self[index]


class Generic:
    def __class_getitem__(cls, params):
        return cls


class Provider:
    BTree = 0
    Hash = 1


def erase_generics(text):
    # Only identifiers naming types/constructors have generic brackets erased.
    names = r'(?:list|array|slice|btree|dict|set|count|deque|heap|vector|allocate)'
    return re.sub(r'\b(' + names + r')\[[^\[\]]*\]', r'\1', text)


def parameters(text):
    parts = re.split(r',\s*(?![^\[]*\])', text)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        name = part.split(':')[0].strip()
        if '...' in part:
            name = '*' + name
        if '=' in part:
            name += '=' + part.split('=', 1)[1].strip()
        result.append(name)
    return ', '.join(result)


def translate(path, env):
    """Translate the Python-shaped subset used by these source algorithms."""
    source = path.read_text()
    source = re.sub(r"(?s)'''(.*?)'''|\"\"\"(.*?)\"\"\"", '', source)
    lines = source.splitlines()
    output = []
    current = None
    fields, methods = set(), set()
    overloads = {}
    skipping = False
    constructor = False
    explicit = path.stem in {"btree", "dict", "set", "storage"}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if line.startswith(('import ', 'trait ', 'enum ', 'test ')):
            skipping = True
            continue
        if line.startswith(('class ', 'def ')):
            skipping = False
        if skipping:
            continue
        if line.startswith('class '):
            current = re.match(r'class (\w+)', line)[1]
            fields, methods = set(), set()
            for body in lines[i + 1:]:
                if body and not body.startswith((' ', '#')):
                    break
                match = re.match(r'    (\w+):', body)
                if match:
                    fields.add(match[1])
                match = re.match(r'    def (\w+)', body)
                if match:
                    methods.add(match[1])
            output.append(f'class {current}(Generic):')
            continue
        if line.startswith('def '):
            current = None
            fields, methods = set(), set()
        if stripped.startswith('trait ') or (len(line) - len(line.lstrip()) == 4 and re.match(r'\w+:', stripped)):
            continue
        if stripped.startswith('operator '):
            if '->' in stripped and not stripped.endswith(':'):
                continue
            name = {'[]': '__getitem__', '[]=': '__setitem__', '+': '__add__', '-': '__sub__', '*': '__mul__', '/': '__truediv__'}.get(stripped.split('(')[0][9:])
            if name is None:
                raise ValueError('unsupported operator: ' + line)
            p = stripped[stripped.index('(') + 1:stripped.index(')')]
            # Slice overload is called explicitly by adapter tests.
            if 'Range[' in p:
                name = 'slice_range'
            if current == 'vector' and name in ('__mul__', '__truediv__') and 'right: T' in p:
                name = 'scale' if name == '__mul__' else 'divide_scalar'
            line = '    def ' + name + '(self, ' + parameters(p) + '):'
        elif stripped.startswith('def '):
            match = re.match(r'(\s*)def (\w+)(?:\[[^]]*\])?\((.*?)\).*:', line)
            if not match:
                raise ValueError('unsupported declaration: ' + line)
            indent, name, args = match.groups()
            p = parameters(args)
            constructor = bool(current and name == current)
            if current and name == current:
                name = '__init__'
                p = 'self' + (', ' + p if p else '')
                key = (current, name)
                index = overloads.get(key, 0)
                overloads[key] = index + 1
                if index:
                    name = 'init_' + str(index)
            elif current and name == 'default':
                output.append(indent + '@staticmethod')
            elif current and not p.startswith('self'):
                p = 'self' + (', ' + p if p else '')
            if current == 'list' and name == 'pop' and 'index' in p:
                name = 'pop_index'
            line = indent + 'def ' + name + '(' + p + '):'
        else:
            line = line.replace(':=', '=').replace('throw ', 'raise ')
            line = re.sub(r'\b(?:move|borrow)\s+', '', line)
            line = line.replace('unsafe:', 'if True:')
            if current and (constructor or not explicit):
                # Resolve implicit receiver fields/methods as Severian does.
                for name in sorted(fields | (methods - {current, 'range'}), key=len, reverse=True):
                    line = re.sub(r'(?<![.\w])' + re.escape(name) + r'\b', 'self.' + name, line)
            line = erase_generics(line)
            line = re.sub(r'\btrue\b', 'True', line)
            line = re.sub(r'\bfalse\b', 'False', line)
        output.append(line)
    generated = '\n'.join(output) + '\n'
    # The restricted adapter does not model privacy; keep names inspectable.
    generated = re.sub(r'\b__(\w+)\b', lambda m: m[0] if m[0].endswith('__') else '_s_' + m[1], generated)
    try:
        exec(compile(generated, str(path) + ':algorithm-adapter', 'exec'), env)
    except Exception:
        Path('/tmp/collection-adapter-failure.py').write_text(generated)
        raise


def load():
    env = dict(Generic=Generic, T=Scalar, K=Scalar, V=Scalar, usize=Scalar,
               u64=Scalar, lossy='lossy', list=Slots, slice=Slice, allocate=Memory, free=free,
               set_storage=Provider, ValueError=ValueError, IndexError=IndexError,
               KeyError=KeyError, range=range, min=min, max=max)
    for filename in ('storage', 'btree', 'dict', 'set'):
        translate(DIRECTORY / (filename + '.sev'), env)
    # Constructors are language overloads; adapters select them by arguments.
    tree = env['btree']
    tree_init = tree.__init__
    def init_tree(self, degree=None):
        if degree is None:
            tree_init(self)
        else:
            self.init_1(degree)
    tree.__init__ = init_tree
    set_type = env['set']
    def init_set(self, provider=Provider.BTree, *values):
        self.init_1(provider, *values)
    set_type.__init__ = init_set
    # Leaf collections are loaded separately so their actual bodies, including
    # allocation, shifting and ring arithmetic, run against instrumented memory.
    leaves = dict(env)
    for filename in ('list', 'deque', 'heap', 'count', 'vector'):
        translate(DIRECTORY / (filename + '.sev'), leaves)
    seq = leaves['list']
    seq_init = seq.__init__
    def init_list(self, values=None):
        if values is None:
            seq_init(self)
        else:
            self.init_2(values)
    seq.__init__ = init_list
    seq_pop = seq.pop
    def pop_list(self, index=None):
        return seq_pop(self) if index is None else self.pop_index(index)
    seq.pop = pop_list
    seq.__iter__ = seq.iter
    dq = leaves['deque']
    dq_init = dq.__init__
    def init_deque(self, capacity=None):
        dq_init(self) if capacity is None else self.init_1(capacity)
    dq.__init__ = init_deque
    dq.__iter__ = dq.iter
    return env, leaves
