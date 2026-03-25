from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

ASSIGNMENT_PATTERN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=.*$")


def read_env_values(path: Path) -> dict[str, str]:
    """Read a dotenv file into a plain string mapping."""

    if not path.exists():
        return {}
    parsed = dotenv_values(path)
    return {key: value for key, value in parsed.items() if value is not None}


def upsert_env_file(path: Path, updates: Mapping[str, str]) -> None:
    """Update or append environment variables while preserving unrelated lines."""

    path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines = path.read_text().splitlines() if path.exists() else []
    output_lines: list[str] = []
    seen_keys: set[str] = set()

    for line in existing_lines:
        match = ASSIGNMENT_PATTERN.match(line)
        if not match:
            output_lines.append(line)
            continue

        key = match.group(1)
        if key not in updates:
            output_lines.append(line)
            continue

        output_lines.append(f"{key}={serialize_env_value(updates[key])}")
        seen_keys.add(key)

    missing_keys = [key for key in updates if key not in seen_keys]
    if missing_keys and output_lines and output_lines[-1].strip():
        output_lines.append("")

    for key in missing_keys:
        output_lines.append(f"{key}={serialize_env_value(updates[key])}")

    if not output_lines:
        output_lines = [f"{key}={serialize_env_value(value)}" for key, value in updates.items()]

    path.write_text("\n".join(output_lines) + "\n")


def serialize_env_value(value: str) -> str:
    """Serialize an env var value, quoting only when shell-safe text is not enough."""

    if value == "":
        return ""

    if re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value):
        return value

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
