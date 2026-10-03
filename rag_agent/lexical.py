from __future__ import annotations

import re


LATIN_TERM_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9_'’-]*|\d+(?:\.\d+)?)(?![A-Za-z0-9_])")
CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")


def lexical_terms(text: str) -> list[str]:
    r"""Return script-aware lexical features in source order."""
    positioned: list[tuple[int, str]] = []
    for match in LATIN_TERM_RE.finditer(text):
        positioned.append((match.start(), match.group(0).casefold()))
    for match in CJK_RUN_RE.finditer(text):
        value = match.group(0)
        features = [value] if len(value) == 1 else [value[index : index + 2] for index in range(len(value) - 1)]
        positioned.extend((match.start() + index, feature) for index, feature in enumerate(features))
    positioned.sort(key=lambda item: item[0])
    return list(dict.fromkeys(term for _, term in positioned))


def fts_index_text(text: str) -> str:
    """Append space-delimited CJK features while preserving the searchable source text."""
    features: list[str] = []
    for match in CJK_RUN_RE.finditer(text):
        value = match.group(0)
        if len(value) == 1:
            features.append(value)
        else:
            features.extend(value[index : index + 2] for index in range(len(value) - 1))
    unique = list(dict.fromkeys(features))
    return f"{text}\n{' '.join(unique)}" if unique else text
