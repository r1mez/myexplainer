from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


def load_yaml_file(path: Union[Path, str]) -> Any:
    """
    Load a small YAML subset used by this repository's config files.

    Supported features:
    - indentation-based mappings
    - lists of scalars or nested blocks
    - scalar values: strings, ints, floats, booleans, null
    - quoted strings with basic escaping
    """
    resolved_path = Path(path)
    lines = _significant_lines(resolved_path.read_text(encoding="utf-8").splitlines())
    if not lines:
        return {}

    payload, next_index = _parse_block(lines, 0, 0)
    if next_index != len(lines):
        raise ValueError(f"Unexpected trailing YAML content in {resolved_path}.")
    return payload


def _significant_lines(lines: List[str]) -> List[str]:
    kept = []
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        kept.append(line)
    return kept


def _parse_block(lines: List[str], index: int, indent: int) -> Tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    current_indent = _indent_of(lines[index])
    if current_indent != indent:
        raise ValueError(
            f"Invalid indentation at line {index + 1}: expected {indent} spaces, got {current_indent}."
        )

    stripped = lines[index][indent:]
    if stripped.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_list(lines: List[str], index: int, indent: int) -> Tuple[List[Any], int]:
    items: List[Any] = []
    while index < len(lines):
        current_indent = _indent_of(lines[index])
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(
                f"Invalid list indentation at line {index + 1}: expected {indent} spaces, got {current_indent}."
            )

        stripped = lines[index][indent:]
        if not stripped.startswith("- "):
            break

        content = stripped[2:].strip()
        index += 1
        if content:
            items.append(_parse_scalar(content))
            continue

        if index >= len(lines) or _indent_of(lines[index]) <= indent:
            items.append(None)
            continue

        nested_value, index = _parse_block(lines, index, indent + 2)
        items.append(nested_value)

    return items, index


def _parse_mapping(lines: List[str], index: int, indent: int) -> Tuple[Dict[str, Any], int]:
    mapping: Dict[str, Any] = {}
    while index < len(lines):
        current_indent = _indent_of(lines[index])
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(
                f"Invalid mapping indentation at line {index + 1}: expected {indent} spaces, got {current_indent}."
            )

        stripped = lines[index][indent:]
        if stripped.startswith("- "):
            raise ValueError(f"Unexpected list item inside mapping at line {index + 1}.")

        if ":" not in stripped:
            raise ValueError(f"Invalid mapping entry at line {index + 1}: {stripped}")

        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        index += 1

        if rest:
            mapping[key] = _parse_scalar(rest)
            continue

        if index >= len(lines) or _indent_of(lines[index]) <= indent:
            mapping[key] = {}
            continue

        nested_value, index = _parse_block(lines, index, indent + 2)
        mapping[key] = nested_value

    return mapping, index


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return _unquote(text)

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    return text


def _unquote(text: str) -> str:
    quote = text[0]
    body = text[1:-1]
    if quote == '"':
        return body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    return body.replace("\\'", "'").replace("\\\\", "\\")
