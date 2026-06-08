"""Minimal YAML load/dump for the restricted experiment-config schema.

Supports: comments (#), nested mappings (2-space indent), scalars
(str/int/float/bool/null), and quoted strings. No flow collections or block
sequences are used by the templates. Avoids a PyYAML dependency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _scalar(v: str) -> Any:
    v = v.strip()
    if v == "" or v.lower() in ("null", "~", "none"):
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if (v[0] == v[-1]) and v[0] in ("'", '"') and len(v) >= 2:
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _strip_comment(line: str) -> str:
    # strip trailing comment not inside quotes (templates have no '#' in values)
    out, q = [], None
    for ch in line:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in ("'", '"'):
            q = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def loads(text: str) -> Dict[str, Any]:
    lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        s = _strip_comment(raw)
        if not s.strip():
            continue
        indent = len(s) - len(s.lstrip(" "))
        lines.append((indent, s.strip()))

    def parse(i: int, indent: int) -> Tuple[Dict[str, Any], int]:
        node: Dict[str, Any] = {}
        while i < len(lines):
            ind, content = lines[i]
            if ind < indent:
                break
            if ind > indent:
                raise ValueError(f"bad indentation: {content!r}")
            if ":" not in content:
                raise ValueError(f"expected key: value, got {content!r}")
            key, _, val = content.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # could be nested map or empty value
                if i + 1 < len(lines) and lines[i + 1][0] > indent:
                    child, i = parse(i + 1, lines[i + 1][0])
                    node[key] = child
                    continue
                node[key] = None
            else:
                node[key] = _scalar(val)
            i += 1
        return node, i

    root, _ = parse(0, 0) if lines else ({}, 0)
    return root


def dumps(obj: Any, indent: int = 0) -> str:
    pad = "  " * indent
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                out.append(f"{pad}{k}:")
                out.append(dumps(v, indent + 1))
            else:
                out.append(f"{pad}{k}: {_fmt(v)}")
    else:
        out.append(f"{pad}{_fmt(obj)}")
    return "\n".join(x for x in out if x != "")


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
