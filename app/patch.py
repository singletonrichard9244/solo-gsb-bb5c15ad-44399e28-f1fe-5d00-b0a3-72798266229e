"""Deterministic RFC 6902 JSON Patch generation.

Given the original ``base_doc`` and the final merge ``result`` of a resolved
merge order, :func:`build_patch` produces the operation sequence that turns
base into result exactly:

  * objects are compared key-by-key and recursed into;
  * a key only on one side becomes ``add`` / ``remove`` carrying the whole
    subtree (added/removed subtrees are not expanded op-by-op);
  * a key on both sides with a different value becomes ``replace``
    (or a deeper operation sequence for nested objects);
  * arrays are treated atomically: any difference emits one ``replace``
    of the whole array (they are never diffed element-by-element);
  * type changes likewise collapse into a single ``replace``.

Operations are sorted by their escaped JSON Pointer path, so the output is
byte-stable for any given (base, result) pair. Path tokens are escaped per
RFC 6901 (``~`` -> ``~0``, ``/`` -> ``~1``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .merge import escape


def _same(a: Any, b: Any) -> bool:
    """Deep equality that never treats ``bool`` as ``int`` (``True`` != 1).

    Plain ``==`` would consider ``1`` and ``True`` equal even through
    nested containers, which would wrongly suppress a replace op.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is bool and type(b) is bool and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _diff(base: Any, result: Any, path: str, ops: list[dict]) -> None:
    # Both sides are objects: recurse key-by-key.
    if isinstance(base, dict) and isinstance(result, dict):
        for key in base.keys() | result.keys():
            token = path + "/" + escape(key)
            if key not in result:
                ops.append({"op": "remove", "path": token})
            elif key not in base:
                ops.append({"op": "add", "path": token, "value": result[key]})
            elif not _same(base[key], result[key]):
                _diff(base[key], result[key], token, ops)
        return
    # Scalars, arrays and type changes are replaced atomically (including
    # the root path, where ``replace`` swaps the whole document).
    ops.append({"op": "replace", "path": path, "value": result})


def build_patch(base: Any, result: Any) -> list[dict]:
    """Return the deterministic RFC 6902 patch transforming base into result.

    Operations are emitted in lexicographic order of their (escaped) JSON
    Pointer path. Applying them in the returned order yields ``result``
    exactly; equal documents yield an empty patch.
    """
    if _same(base, result):
        return []
    ops: list[dict] = []
    _diff(base, result, "", ops)
    ops.sort(key=lambda op: op["path"])
    return ops


def canonical_json(doc: Any) -> bytes:
    """Canonical JSON serialization used for the SHA-256 digests.

    Keys are sorted and compact separators are used, making the digest
    independent of insertion order or formatting.
    """
    return json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256(doc: Any) -> str:
    """SHA-256 hex digest of a document's canonical JSON encoding."""
    return hashlib.sha256(canonical_json(doc)).hexdigest()
