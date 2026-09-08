#!/usr/bin/env python3
"""Differential tests of .sev algorithm bodies through the restricted adapter.

Passing this suite is algorithm evidence only, never native/ownership acceptance.
Run collection_capabilities.py separately for the actual compiler stages.
"""
from collections import Counter, deque
import random
import unittest

from collection_source_adapter import Key, Memory, Scalar, Slice, Slots, load


class Collision(Key):
    def hash(self):
        return 3


class Collections(unittest.TestCase):
    def setUp(self):
        self.env, self.leaves = load()

    def tree_invariants(self, tree, model):
        self.assertEqual(tree.len(), len(model))
        self.assertEqual(list(tree.items()), sorted(model.items()))
        self.assertEqual(list(tree.keys()), sorted(model))
        seen, depths = set(), set()
        total = 0
        def visit(node, lower, upper, depth):
            nonlocal total
            self.assertNotIn(node, seen)
            seen.add(node)
            metadata = tree._s_nodes[node]
            self.assertLessEqual(metadata.count, tree._s_width)
            self.assertGreaterEqual(metadata.count, 1 if node == tree.root else tree._s_degree - 1)
            keys = [tree._s_key(node, i) for i in range(metadata.count)]
            self.assertEqual(keys, sorted(set(keys)))
            for key in keys:
                self.assertTrue(lower is None or lower < key)
                self.assertTrue(upper is None or key < upper)
            total += len(keys)
            if metadata.leaf:
                depths.add(depth)
            else:
                bounds = [lower, *keys, upper]
                for index in range(len(keys) + 1):
                    visit(tree._s_child(node, index), bounds[index], bounds[index + 1], depth + 1)
        if model:
            visit(tree.root, None, None, 0)
            self.assertEqual(len(depths), 1)
        self.assertEqual(total, len(model))
        free = list(tree._s_free)
        self.assertEqual(len(free), len(set(free)))
        self.assertEqual(seen | set(free), set(range(tree._s_nodes.len())))
        self.assertFalse(seen & set(free))

    def test_btree_random_mutations_all_invariants(self):
        for degree in (2, 3, 8):
            tree = self.env['btree'](degree)
            model = {}
            rng = random.Random(100 + degree)
            for step in range(1600):
                key = rng.randrange(180)
                if rng.randrange(3):
                    value = rng.randrange(10000)
                    self.assertEqual(tree.insert(key, value), (model.get(key, 0), key in model))
                    model[key] = value
                else:
                    expected = (model.get(key, 0), key in model)
                    self.assertEqual(tree.remove(key), expected)
                    model.pop(key, None)
                self.tree_invariants(tree, model)
                self.assertEqual(tree.get(key), (model.get(key, 0), key in model))
                ordered = sorted(model)
                for strict, method in ((False, tree.lower_bound), (True, tree.upper_bound)):
                    eligible = [k for k in ordered if k > key or (not strict and k == key)]
                    expected = (eligible[0], model[eligible[0]], True) if eligible else (0, 0, False)
                    self.assertEqual(method(key), expected)
                stop = key + rng.randrange(25)
                self.assertEqual(list(tree.range(key, stop)), [(k, model[k]) for k in ordered if key <= k < stop])
            keys = list(model)
            rng.shuffle(keys)
            for key in keys:
                self.assertEqual(tree.remove(key), (model.pop(key), True))
                self.tree_invariants(tree, model)
            tree.insert(42, 9)
            self.tree_invariants(tree, {42: 9})
            tree.clear()
            self.tree_invariants(tree, {})

    def test_btree_extremes_copy_replacement_and_degree_validation(self):
        for degree in (0, 1, Scalar.max // 4 + 1, Scalar.max):
            with self.assertRaises(ValueError):
                self.env['btree'](degree)
        tree = self.env['btree'](2)
        self.assertEqual(tree.first(), (0, 0, False))
        self.assertEqual(tree.last(), (0, 0, False))
        for key in range(100):
            tree.set(key, key * 10)
        self.assertEqual(tree.first(), (0, 0, True))
        self.assertEqual(tree.last(), (99, 990, True))
        other = tree.copy()
        other.clear()
        self.assertEqual(tree.len(), 100)
        self.assertEqual(list(tree.range(40, 40)), [])
        self.assertEqual(list(tree.range(90, 10)), [])
        self.assertEqual(tree.replace(5, 3), (50, True))
        self.assertEqual(tree[5], 3)
        with self.assertRaises(KeyError):
            tree[1000]

    def dict_invariants(self, table, model):
        self.assertEqual(table.len(), len(model))
        self.assertEqual(list(table.items()), list(model.items()))
        self.assertEqual(list(table.keys()), list(model))
        self.assertEqual(list(table.values()), list(model.values()))
        live = [entry - 2 for entry in table._s_buckets if entry > 1]
        self.assertEqual(len(live), len(set(live)))
        self.assertEqual(len(live), len(model))
        self.assertLessEqual(len(live) * 2, table._s_buckets.len())
        self.assertEqual(table._s_used, sum(entry != 0 for entry in table._s_buckets))
        self.assertEqual(set(live) | set(table._s_free), set(range(table._s_keys.len())))
        self.assertFalse(set(live) & set(table._s_free))
        previous, entry, seen = 0, table._s_head, set()
        while entry:
            self.assertNotIn(entry, seen)
            seen.add(entry)
            self.assertEqual(table._s_previous[entry - 1], previous)
            previous, entry = entry, table._s_next[entry - 1]
        self.assertEqual(previous, table._s_tail)
        self.assertEqual({entry - 1 for entry in seen}, set(live))
        for key, value in model.items():
            self.assertEqual(table.get(key), (value, True))

    def test_hash_collisions_rehash_deletion_reuse_and_order(self):
        for key_type in (Key, Collision):
            table = self.env['dict']()
            model = {}
            rng = random.Random(42)
            for step in range(1000):
                key = key_type(rng.randrange(90))
                op = rng.randrange(5)
                if op < 2:
                    value = rng.randrange(1000)
                    self.assertEqual(table.replace(key, value), (model.get(key, 0), key in model))
                    model[key] = value
                elif op == 2:
                    self.assertEqual(table.remove(key), (model.get(key, 0), key in model))
                    model.pop(key, None)
                elif op == 3:
                    self.assertEqual(table.set_default(key, step), model.setdefault(key, step))
                else:
                    table.reserve(rng.randrange(8))
                self.dict_invariants(table, model)
            other = table.copy()
            other.clear()
            self.dict_invariants(table, model)
            slots = table._s_keys.len()
            for key in list(model):
                table.remove(key)
            for key in model:
                table.set(key, model[key])
            self.assertEqual(table._s_keys.len(), slots)
            self.dict_invariants(table, model)
            table.clear()
            self.dict_invariants(table, {})
            table.set(Key(1), 2)
            self.dict_invariants(table, {Key(1): 2})

    def test_dictionary_churn_does_not_rehash_each_mutation(self):
        table = self.env['dict']()
        for key in range(128):
            table.set(Key(key), key)
        rehashes = 0
        rehash = table._s_rehash
        def measured_rehash(size):
            nonlocal rehashes
            rehashes += 1
            rehash(size)
        table._s_rehash = measured_rehash
        for key in range(1000):
            table.remove(Key(key))
            table.set(Key(key + 128), key)
        self.assertLess(rehashes, 32)
        self.assertEqual(table.len(), 128)
        self.assertEqual(table._s_keys.len(), 128)

    def test_set_providers_algebra_order_and_bounds(self):
        for left_provider in (0, 1):
            for right_provider in (0, 1):
                left = self.env['set'](left_provider)
                right = self.env['set'](right_provider)
                for value in (9, 1, 5, 3, 5):
                    left.add(Collision(value))
                for value in (5, 2, 9, 7):
                    right.add(Collision(value))
                self.assertEqual(list(left.iter()), [1, 3, 5, 9] if left_provider == 0 else [9, 1, 5, 3])
                a, b = {1, 3, 5, 9}, {2, 5, 7, 9}
                for operation, expected in (('union', a | b), ('intersection', a & b),
                                            ('difference', a - b), ('symmetric_difference', a ^ b)):
                    result = self.env[operation](left, right)
                    self.assertEqual(result.storage, left_provider)
                    self.assertEqual(set(result.iter()), expected)
                self.assertFalse(self.env['subset'](left, right))
                self.assertTrue(self.env['superset'](left, left))
                self.assertFalse(self.env['disjoint'](left, right))
                self.assertEqual(left.first(), (1, True))
                self.assertEqual(left.last(), (9, True))
                self.assertEqual(left.lower_bound(5), (5, True))
                self.assertEqual(left.upper_bound(5), (9, True))
                self.assertEqual(left.upper_bound(9), (0, False))
                self.assertEqual(list(left.range(2, 9)), [3, 5])
                self.assertEqual(left.pop_first(), (1, True))
                self.assertEqual(left.pop_last(), (9, True))
                self.assertEqual(left.len(), 2)
                left.clear()
                self.assertEqual(left.pop_last(), (0, False))

    def test_stored_key_representatives_are_retained(self):
        for provider in (0, 1):
            first, equal = Collision(7), Collision(7)
            first.label, equal.label = 'original', 'replacement'
            values = self.env['set'](provider)
            values.add(first)
            existing, added = values.add(equal)
            self.assertFalse(added)
            self.assertEqual(existing.label, 'original')
            self.assertEqual(values.get(equal)[0].label, 'original')
            self.assertEqual(values.remove(equal)[0].label, 'original')

    def test_list_shifts_alias_extension_and_copy(self):
        seq = self.leaves['list']()
        model = []
        rng = random.Random(89)
        for _ in range(700):
            op = rng.randrange(5)
            if op == 0:
                value = rng.randrange(100)
                seq.append(value)
                model.append(value)
            elif op == 1:
                index = rng.randrange(len(model) + 1)
                seq.insert(index, 20)
                model.insert(index, 20)
            elif op == 2:
                index = rng.randrange(len(model) + 1)
                self.assertEqual(seq.pop(index), (model.pop(index), True) if index < len(model) else (0, False))
            elif op == 3:
                seq.reverse()
                model.reverse()
            else:
                new_length = rng.randrange(len(model) + 2)
                seq.truncate(new_length)
                model = model[:new_length]
            self.assertEqual(list(seq), model)
            self.assertGreaterEqual(seq.cap(), seq.len())
        seq.clear()
        for value in (1, 2, 3, 4):
            seq.append(value)
        borrowed = Slice(seq._s_memory, 1, 3)
        seq.extend(borrowed)
        self.assertEqual(list(seq), [1, 2, 3, 4, 2, 3, 4])
        other = seq.copy()
        other[0] = 99
        self.assertEqual(seq[0], 1)
        self.assertEqual(seq.find(3), (2, True))
        self.assertEqual(seq.remove(3), (3, True))
        self.assertEqual(seq.remove(99), (0, False))
        seq.swap(0, 1)
        self.assertEqual(seq[0], 2)
        for action in (lambda: seq[seq.len()], lambda: seq.insert(seq.len() + 1, 2), lambda: seq.swap(0, seq.len())):
            with self.assertRaises(IndexError):
                action()
        seq.drop()
        other.drop()

    def test_deque_wrap_growth_against_reference(self):
        for capacity in (0, 1, 3, 8):
            seq = self.leaves['deque'](capacity)
            model = deque()
            rng = random.Random(23)
            for _ in range(1400):
                op = rng.randrange(7)
                if op == 0:
                    seq.push_front(10)
                    model.appendleft(10)
                elif op < 3:
                    seq.push_back(20)
                    model.append(20)
                elif op == 3:
                    self.assertEqual(seq.pop_front(), (model.popleft(), True) if model else (0, False))
                elif op == 4:
                    self.assertEqual(seq.pop_back(), (model.pop(), True) if model else (0, False))
                elif op == 5 and model:
                    index = rng.randrange(len(model))
                    seq[index] = 30
                    model[index] = 30
                elif op == 6:
                    seq.reserve(4)
                self.assertEqual(list(seq), list(model))
                self.assertEqual(seq.front(), (model[0], True) if model else (0, False))
                self.assertEqual(seq.back(), (model[-1], True) if model else (0, False))
            other = seq.copy()
            seq.clear()
            self.assertEqual(list(other), list(model))
            seq.push_front(42)
            self.assertEqual(list(seq), [42])
            seq.drop()
            other.drop()

    def test_heap_duplicates_replacement_and_push_pop(self):
        import heapq
        heap = self.leaves['heap']()
        model = []
        rng = random.Random(10)
        for _ in range(1800):
            value = rng.randrange(40)
            op = rng.randrange(4)
            if op == 0:
                heap.push(value)
                heapq.heappush(model, value)
            elif op == 1:
                self.assertEqual(heap.pop(), (heapq.heappop(model), True) if model else (0, False))
            elif op == 2:
                if model:
                    self.assertEqual(heap.replace(value), (heapq.heapreplace(model, value), True))
                else:
                    self.assertEqual(heap.replace(value), (0, False))
                    heapq.heappush(model, value)
            else:
                self.assertEqual(heap.push_pop(value), heapq.heappushpop(model, value))
            self.assertEqual(heap.len(), len(model))
            self.assertEqual(heap.peek(), (model[0], True) if model else (0, False))
        heap.clear()
        self.assertEqual(heap.len(), 0)

    def test_count_model_zero_removal_and_overflow_rollback(self):
        counter = self.leaves['count']()
        model = Counter()
        rng = random.Random(13)
        for _ in range(1200):
            key, amount, op = Key(rng.randrange(20)), rng.randrange(8), rng.randrange(3)
            if op == 0:
                counter.add(key, amount)
                if amount:
                    model[key] += amount
            elif op == 1:
                remaining = max(0, model[key] - amount)
                self.assertEqual(counter.subtract(key, amount), remaining)
                if remaining:
                    model[key] = remaining
                else:
                    model.pop(key, None)
            else:
                self.assertEqual(counter.remove(key), model.pop(key, 0))
            self.assertEqual(dict(counter.items()), dict(model))
            self.assertEqual(counter.total(), sum(model.values()))
            self.assertEqual(counter.len(), len(model))
            self.assertEqual(Counter(counter.elements()), model)
        other = counter.copy()
        counter.clear()
        self.assertEqual(dict(other.items()), dict(model))
        counter.add(Key(1), Scalar.max)
        for key in (Key(1), Key(2)):
            with self.assertRaises(ValueError):
                counter.add(key, 1)
            self.assertEqual(counter.total(), Scalar.max)
            self.assertEqual(counter.len(), 1)
            self.assertEqual(counter.frequency(Key(1)), Scalar.max)

    def test_vector_lanes_reductions_mapping_and_copy(self):
        for lanes in (0, 1, 4):
            self.leaves['N'] = lanes
            self.leaves['array'] = lambda: Slots([0] * lanes)
            vector = self.leaves['vector']
            left, right = vector(), vector()
            for index in range(lanes):
                left[index], right[index] = index + 1, index + 2
            self.assertEqual(left.len(), lanes)
            self.assertEqual(left.sum(), sum(range(1, lanes + 1)))
            self.assertEqual(left.dot(right), sum((i + 1) * (i + 2) for i in range(lanes)))
            self.assertEqual(left.length_squared(), sum((i + 1) ** 2 for i in range(lanes)))
            self.assertEqual(list((left + right).iter()), [2 * i + 3 for i in range(lanes)])
            self.assertEqual(list((left - right).iter()), [-1] * lanes)
            self.assertEqual(list((left * right).iter()), [(i + 1) * (i + 2) for i in range(lanes)])
            self.assertEqual(list(left.scale(2).iter()), [2 * (i + 1) for i in range(lanes)])
            self.assertEqual(list(left.map(lambda x: float(x) + 0.5).iter()), [i + 1.5 for i in range(lanes)])
            copied = left.copy()
            if lanes:
                copied[0] = 99
                self.assertEqual(left[0], 1)

    def test_checked_growth(self):
        size, capacity = self.env['collection_size'], self.env['collection_capacity']
        self.assertEqual(size(Scalar.max, 0), Scalar.max)
        with self.assertRaises(ValueError):
            size(Scalar.max, 1)
        self.assertEqual(capacity(0, 0), 0)
        self.assertEqual(capacity(0, 1), 4)
        self.assertEqual(capacity(8, 9), 16)
        self.assertEqual(capacity(Scalar.max // 2 + 1, Scalar.max), Scalar.max)


if __name__ == '__main__':
    unittest.main()
