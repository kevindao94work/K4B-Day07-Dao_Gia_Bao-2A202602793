"""Section-aware Markdown chunking for the CP5 policy corpus."""
from __future__ import annotations

import re

from .chunking import RecursiveChunker


class HeadingChunker:
    """Keep each section together, repeating its heading path on split pieces.

    chunk_size includes headings. Fenced code blocks do not create sections.
    A heading path that alone fills the budget raises ValueError rather than
    silently dropping titles or exceeding the configured size.
    """

    def __init__(self, chunk_size: int = 800) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        chunks: list[str] = []
        headings: list[tuple[int, str]] = []
        body: list[str] = []
        fence = None
        last_emitted_headings = None

        def flush() -> None:
            nonlocal last_emitted_headings
            content = ''.join(body).strip()
            body.clear()
            if not content:
                return
            prefix = '\n'.join(heading for _, heading in headings)
            if prefix:
                prefix += '\n\n'
            budget = self.chunk_size - len(prefix)
            if budget <= 0:
                raise ValueError('Heading path leaves no space for section content')
            for piece in RecursiveChunker(chunk_size=budget).chunk(content):
                if piece.strip():
                    chunks.append(prefix + piece.strip())
            last_emitted_headings = list(headings)

        for line in text.splitlines(keepends=True):
            marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line.rstrip('\n'))
            if marker:
                run, suffix = marker.groups()
                if fence is None:
                    fence = (run[0], len(run))
                elif run[0] == fence[0] and len(run) >= fence[1] and not suffix.strip():
                    fence = None
                body.append(line)
                continue
            match = re.match(r'^ {0,3}(#{1,6})\s+(.+?)\s*$', line) if fence is None else None
            if match:
                flush()
                level = len(match.group(1))
                headings = [(depth, title) for depth, title in headings if depth < level]
                headings.append((level, line.strip()))
            else:
                body.append(line)
        flush()
        # Preserve a document consisting only of headings, or a final empty section.
        if headings and headings != last_emitted_headings:
            title = '\n'.join(heading for _, heading in headings)
            if len(title) > self.chunk_size:
                raise ValueError('Heading path exceeds chunk_size')
            chunks.append(title)
        return chunks
