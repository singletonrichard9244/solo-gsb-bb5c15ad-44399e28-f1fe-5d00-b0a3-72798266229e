"""Unit tests for deterministic RFC 6902 patch generation."""

import copy
import json
import random
import unittest

from app.patch import build_patch, canonical_json, sha256


def apply_patch(doc, ops):
    """Minimal RFC 6902 applier (object keys + array indices) for assertions."""

    def resolve(node, token, create=False):
        if isinstance(node, dict):
            return node, token
        idx = int(token)
        if create and idx == len(node):
            return node, idx
        return node, idx

    for op in ops:
        tokens = op["path"].split("/")[1:] if op["path"] else []
        tokens = [t.replace("~1", "/").replace("~0", "~") for t in tokens]
        if not tokens:
            doc = op.get("value")
            continue
        parent = doc
        for tok in tokens[:-1]:
            parent = parent[tok if isinstance(parent, dict) else int(tok)]
        target, key = resolve(
            parent, tokens[-1], create=(op["op"] == "add")
        )
        if op["op"] == "add":
            target.insert(key, op["value"]) if isinstance(target, list) else target.__setitem__(key, op["value"])
        elif op["op"] == "remove":
            target.pop(key) if isinstance(target, dict) else target.__delitem__(key)
        elif op["op"] == "replace":
            target[key] = op["value"]
        else:  # pragma: no cover - only the three ops are generated
            raise AssertionError(op)
    return doc


def assert_roundtrip(testcase, base, result):
    ops = build_patch(copy.deepcopy(base), copy.deepcopy(result))
    testcase.assertEqual(apply_patch(copy.deepcopy(base), ops), result)
    # Paths are escaped JSON Pointers and sorted lexicographically.
    paths = [op["path"] for op in ops]
    testcase.assertEqual(paths, sorted(paths))
    return ops


class BuildPatchTest(unittest.TestCase):
    def test_equal_documents_emit_empty_patch(self):
        doc = {"a": [1, 2], "b": {"c": "x"}}
        self.assertEqual(build_patch(doc, copy.deepcopy(doc)), [])

    def test_add_remove_replace_at_top_level(self):
        base = {"a": 1, "b": 2}
        result = {"b": 3, "c": 4}
        ops = assert_roundtrip(self, base, result)
        self.assertEqual(
            ops,
            [
                {"op": "remove", "path": "/a"},
                {"op": "replace", "path": "/b", "value": 3},
                {"op": "add", "path": "/c", "value": 4},
            ],
        )

    def test_nested_object_recursion(self):
        base = {"o": {"a": 1, "b": 2, "c": 3}}
        result = {"o": {"a": 1, "b": 20, "d": 4}}
        ops = assert_roundtrip(self, base, result)
        self.assertEqual(
            ops,
            [
                {"op": "replace", "path": "/o/b", "value": 20},
                {"op": "remove", "path": "/o/c"},
                {"op": "add", "path": "/o/d", "value": 4},
            ],
        )

    def test_arrays_replaced_wholesale(self):
        base = {"l": [1, 2, 3], "kept": 1}
        result = {"l": [1, 2, 3, 4], "kept": 1}
        ops = assert_roundtrip(self, base, result)
        self.assertEqual(
            ops, [{"op": "replace", "path": "/l", "value": [1, 2, 3, 4]}]
        )

    def test_escaped_tokens_in_paths(self):
        base = {"a/b": 1, "c~d": {"x": 1}}
        result = {"a/b": 2, "c~d": {"x": 2}, "e/f": {"g~h": 9}}
        ops = assert_roundtrip(self, base, result)
        self.assertEqual(
            ops,
            [
                {"op": "replace", "path": "/a~1b", "value": 2},
                {"op": "replace", "path": "/c~0d/x", "value": 2},
                {"op": "add", "path": "/e~1f", "value": {"g~h": 9}},
            ],
        )
        # Sorting uses the escaped form: '~1' (0x7e...) sorts before digits.
        paths = [op["path"] for op in ops]
        self.assertEqual(paths, sorted(paths))

    def test_type_change_is_single_replace(self):
        base = {"a": {"b": 1}, "n": 1}
        result = {"a": [1, 2], "n": True}
        ops = assert_roundtrip(self, base, result)
        self.assertEqual(
            ops,
            [
                {"op": "replace", "path": "/a", "value": [1, 2]},
                {"op": "replace", "path": "/n", "value": True},
            ],
        )

    def test_root_scalar_replace(self):
        ops = assert_roundtrip(self, {"a": 1}, [1, 2])
        self.assertEqual(ops, [{"op": "replace", "path": "", "value": [1, 2]}])
        ops = assert_roundtrip(self, 5, {"x": 1})
        self.assertEqual(ops, [{"op": "replace", "path": "", "value": {"x": 1}}])

    def test_null_add_and_remove(self):
        self.assertEqual(
            build_patch({}, {"a": None}),
            [{"op": "add", "path": "/a", "value": None}],
        )
        self.assertEqual(
            build_patch({"a": None}, {}),
            [{"op": "remove", "path": "/a"}],
        )

    def test_bool_distinguished_from_int(self):
        # True must not be considered equal to 1.
        ops = assert_roundtrip(self, {"a": 1}, {"a": True})
        self.assertEqual(ops, [{"op": "replace", "path": "/a", "value": True}])

    def test_deterministic_across_key_insertion_order(self):
        base = {"z": 1, "a": {"y": 1, "b": 2}}
        result = {"a": {"b": 3, "y": 1}, "z": 1, "new": [9]}
        ops1 = build_patch(copy.deepcopy(base), copy.deepcopy(result))
        result2 = {"new": [9], "a": {"b": 3, "y": 1}, "z": 1}
        ops2 = build_patch(copy.deepcopy(base), result2)
        self.assertEqual(ops1, ops2)
        assert_roundtrip(self, base, result)

    def test_sha256_uses_canonical_json(self):
        self.assertEqual(
            sha256({"a": 1, "b": 2}), sha256({"b": 2, "a": 1})
        )
        self.assertEqual(
            sha256({"a": 1, "b": 2}),
            __import__("hashlib").sha256(b'{"a":1,"b":2}').hexdigest(),
        )
        self.assertNotEqual(sha256({"a": 1}), sha256({"a": 2}))
        self.assertEqual(
            json.loads(canonical_json({"b": 1, "a": 2})), {"b": 1, "a": 2}
        )

    def test_randomized_roundtrips(self):
        rng = random.Random(42)

        def gen(depth=0):
            kind = rng.randrange(6 if depth < 3 else 3)
            if kind == 0:
                return rng.randrange(100)
            if kind == 1:
                return rng.choice(["x", "a/b", "c~d", ""])
            if kind == 2:
                return rng.choice([True, False, None])
            if kind == 3:
                return [gen(depth + 1) for _ in range(rng.randrange(4))]
            return {
                rng.choice(["k", "a/b", "c~d", "n"]): gen(depth + 1)
                for _ in range(rng.randrange(5))
            }

        for _ in range(500):
            assert_roundtrip(self, gen(), gen())


if __name__ == "__main__":
    unittest.main()
