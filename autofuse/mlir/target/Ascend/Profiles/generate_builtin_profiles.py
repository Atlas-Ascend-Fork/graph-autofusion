#!/usr/bin/env python3
"""Embed the selected CANN platform INI files as read-only C++ resources."""

from pathlib import Path
import re
import sys

MAX_PLATFORM_NAME_LENGTH = 128


def cpp_string(value: bytes) -> str:
    lines = []
    for line in value.splitlines(keepends=True):
        escaped = line.decode("utf-8").replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\r", "\\r").replace("\n", "\\n")
        lines.append(f'      "{escaped}"')
    if not lines:
        return '      ""'
    return "\n".join(lines)


def is_platform_name(value: str) -> bool:
    return (
        re.fullmatch(rf"[A-Za-z0-9_-]{{1,{MAX_PLATFORM_NAME_LENGTH}}}", value)
        is not None
    )


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} MANIFEST PROFILE_DIR OUTPUT", file=sys.stderr)
        return 2

    manifest = Path(sys.argv[1])
    profile_dir = Path(sys.argv[2])
    output = Path(sys.argv[3])
    groups = []
    seen_aliases = set()

    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"{manifest}:{line_number}: expected canonical:alias|alias"
            )
        canonical, alias_list = line.split(":", 1)
        aliases = alias_list.split("|")
        if not is_platform_name(canonical) or any(
            not is_platform_name(alias) for alias in aliases
        ):
            raise ValueError(f"{manifest}:{line_number}: invalid platform name")
        if not canonical or not aliases or aliases[0] != canonical:
            raise ValueError(
                f"{manifest}:{line_number}: canonical platform must be the first alias"
            )
        if any(not alias or alias in seen_aliases for alias in aliases):
            raise ValueError(
                f"{manifest}:{line_number}: duplicate or empty platform alias"
            )
        ini = profile_dir / f"{canonical}.ini"
        if not ini.is_file():
            raise ValueError(f"{manifest}:{line_number}: missing canonical INI {ini}")
        seen_aliases.update(aliases)
        groups.append((canonical, aliases, ini.read_bytes()))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("// Generated from CANN platform INI files. Do not edit.\n")
        stream.write(
            "constexpr BuiltinProfileResource kBuiltinProfileResources[] = {\n"
        )
        for canonical, aliases, ini in groups:
            stream.write("  {\n")
            stream.write(f'      "{canonical}",\n')
            stream.write(f'      "{",".join(aliases)}",\n')
            stream.write("      llvm::StringLiteral(\n")
            stream.write(cpp_string(ini))
            stream.write("),\n  },\n")
        stream.write("};\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
