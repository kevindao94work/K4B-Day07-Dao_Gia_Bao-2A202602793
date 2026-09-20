#!/usr/bin/env python3
"""CP5: shared corpus/query runner; change only CHUNKER for a fair comparison.

Default: real multilingual local embeddings. --provider mock is a plumbing check
only and is explicitly labeled. No silent fallback to mock and no paid API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from importlib.metadata import PackageNotFoundError, version
import re
from datetime import datetime, timezone
from pathlib import Path

from src.chunking import ChunkingStrategyComparator, FixedSizeChunker, RecursiveChunker
from src.embeddings import LOCAL_EMBEDDING_MODEL, LocalEmbedder, MockEmbedder
from src.heading_chunker import HeadingChunker
from src.models import Document
from src.store import EmbeddingStore

ROOT = Path(__file__).resolve().parent
CHUNKER = HeadingChunker(chunk_size=800)  # Change only this line for your strategy.
# Alternatives: FixedSizeChunker(chunk_size=800, overlap=50), RecursiveChunker(chunk_size=800)
BASELINE_DOCS = ['tiki-doi-tra-365', 'tiki-dropship-doi-tra-bao-hanh', 'tiki-ngon-doi-tra-boi-thuong']
REQUIRED = {'doc_id', 'title', 'source_url', 'retrieved_at', 'document_version', 'audience'}


def installed_package_versions() -> dict[str, str]:
    packages = {}
    for name in ['sentence-transformers', 'transformers', 'torch']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'not-installed-in-current-interpreter'
    return packages


def make_chunker(strategy: str):
    """Return a comparable CP5 chunker; default preserves the personal choice."""
    if strategy == 'selected':
        return CHUNKER
    if strategy == 'fixed':
        return FixedSizeChunker(chunk_size=800, overlap=50)
    if strategy == 'recursive':
        return RecursiveChunker(chunk_size=800)
    if strategy == 'heading':
        return HeadingChunker(chunk_size=800)
    raise ValueError(f'Unknown strategy: {strategy}')


def load_markdown(path: Path) -> tuple[dict, str]:
    """Read the flat, JSON-quoted YAML strings emitted by the course crawler.

    Reject unsupported YAML instead of silently losing metadata; this is not a
    general YAML parser. The CP2 corpus uses precisely this front matter format.
    """
    match = re.match(r'\A---\s*\n(.*?)\n---\s*\n(.*)\Z', path.read_text(encoding='utf-8'), re.S)
    if not match:
        raise ValueError(f'{path}: missing front matter')
    metadata = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(':')
        if not separator or not re.fullmatch(r'[a-z][a-z0-9_]*', key) or key in metadata:
            raise ValueError(f'{path}: invalid or duplicate metadata key')
        metadata[key] = json.loads(value.strip())
        if not isinstance(metadata[key], str):
            raise ValueError(f'{path}: metadata must be string-valued')
    if not REQUIRED <= metadata.keys() or metadata['doc_id'] != path.stem:
        raise ValueError(f'{path}: missing metadata or mismatched doc_id')
    body = match.group(2).strip()
    if not body or metadata['audience'] not in {'buyer', 'seller', 'both'}:
        raise ValueError(f'{path}: empty body or invalid audience')
    return metadata, body


class CachedEmbedder:
    """Content-hash cache namespaced by provider, model and normalization."""

    def __init__(self, provider: str, model: str, directory: Path) -> None:
        self.provider, self.model, self.directory = provider, model, directory
        self.backend = None
        self.hits = self.misses = 0
        self.namespace = f'cp5-v1:{provider}:{model}:unit-norm'

    def __call__(self, text: str) -> list[float]:
        key = hashlib.sha256((self.namespace + '\0' + text).encode('utf-8')).hexdigest()
        path = self.directory / f'{key}.json'
        if path.exists():
            vector = json.loads(path.read_text(encoding='utf-8'))
            self.hits += 1
        else:
            if self.backend is None:
                print(f'Loading embedding backend: {self.provider} / {self.model}', flush=True)
                self.backend = LocalEmbedder(self.model) if self.provider == 'local' else MockEmbedder()
            vector = self.backend(text)
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(vector), encoding='utf-8')
            temporary.replace(path)
            self.misses += 1
        if not vector or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in vector):
            raise ValueError(f'Invalid cached embedding: {path}')
        if not math.isclose(sum(x*x for x in vector), 1.0, abs_tol=1e-4):
            raise ValueError(f'Embedding is not unit normalized: {path}')
        return vector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['local', 'mock'], default='local')
    parser.add_argument('--model', default=LOCAL_EMBEDDING_MODEL)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'data/tiki-doi-tra')
    parser.add_argument('--queries', type=Path, default=ROOT / 'data/benchmark_queries.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'report/cp5')
    parser.add_argument('--output', type=Path, help='Write a readable top-3 benchmark log to this file.')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.cache/bench-embeddings')
    parser.add_argument('--strategy', choices=['selected', 'fixed', 'recursive', 'heading'],
                        default='selected', help='Use selected for the personal CHUNKER line.')
    args = parser.parse_args()
    pages = [(path, *load_markdown(path)) for path in sorted(args.corpus.glob('*.md'))]
    if not pages:
        parser.error('Corpus contains no Markdown documents')
    query_bytes = args.queries.read_bytes()
    queries = json.loads(query_bytes)
    corpus_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _ in pages}
    if len(queries) != 5 or len({q['id'] for q in queries}) != 5:
        parser.error('Exactly five uniquely identified queries are required')
    if not any(q['metadata_filter'].get('audience') in {'buyer', 'seller'} for q in queries):
        parser.error('At least one query must filter by audience')
    source_docs = {metadata['doc_id']: (metadata, body) for _, metadata, body in pages}
    for q in queries:
        metadata, body = source_docs[q['source_doc_id']]
        if q['evidence_quote'] not in body:
            raise ValueError(f"{q['id']}: evidence missing from source")
        if not all(metadata.get(k) == v for k, v in q['metadata_filter'].items()):
            raise ValueError(f"{q['id']}: filter excludes gold document")

    chunker = make_chunker(args.strategy)
    docs = []
    counts = {}
    for path, metadata, body in pages:
        chunks = chunker.chunk(body)
        counts[path.stem] = len(chunks)
        docs.extend(Document(id=f'{path.stem}#{i}', content=chunk,
                             metadata={**metadata, 'doc_id': path.stem,
                                       'chunk_index': i, 'file_path': str(path.relative_to(ROOT))
                                       if path.is_relative_to(ROOT) else str(path)})
                    for i, chunk in enumerate(chunks))
    model = args.model if args.provider == 'local' else 'MockEmbedder-64'
    embedder = CachedEmbedder(args.provider, model, args.cache_dir)
    print(f'Strategy: {type(chunker).__name__} {vars(chunker)}', flush=True)
    print(f'Embedding: {args.provider} / {model}', flush=True)
    if args.provider == 'mock':
        print('MOCK: plumbing check only; scores do not measure semantic relevance.', flush=True)
    store = EmbeddingStore('cp5', embedding_fn=embedder)
    store.add_documents(docs)
    print(f'Loaded {len(pages)} files -> {store.get_collection_size()} chunks', flush=True)
    results = []
    for q in queries:
        hits = store.search_with_filter(q['query'], top_k=3, metadata_filter=q['metadata_filter'])
        unfiltered = store.search(q['query'], top_k=3) if q['metadata_filter'] else hits
        evidence_ids = [d.id for d in docs if d.metadata['doc_id'] == q['source_doc_id']
                        and q['evidence_quote'] in d.content]
        print(f"\n{q['id']}: {q['query']}\nfilter={json.dumps(q['metadata_filter'])}")
        print(f"Gold: {q['gold_answer']}\nSource: {q['source_doc_id']} / {q['source_heading']}")
        for rank, hit in enumerate(hits, 1):
            print(f"  {rank}. score={hit['score']:.6f} doc_id={hit['metadata']['doc_id']} "
                  f"chunk_id={hit['id']} audience={hit['metadata']['audience']}")
            print('     ' + hit['content'].replace('\n', ' ')[:240])
        if q['metadata_filter']:
            print('  Without filter: ' + ', '.join(f"{h['id']} ({h['metadata']['audience']}, {h['score']:.6f})" for h in unfiltered))
        results.append({**q, 'evidence_chunk_ids': evidence_ids, 'top3': hits, 'unfiltered_top3': unfiltered})

    baseline = {doc_id: ChunkingStrategyComparator().compare(source_docs[doc_id][1], chunk_size=800)
                for doc_id in BASELINE_DOCS if doc_id in source_docs}
    output = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'strategy': type(chunker).__name__, 'strategy_parameters': vars(chunker),
        'embedding_provider': args.provider, 'embedding_model': model,
        'semantic_embeddings': args.provider != 'mock',
        'corpus_sha256': corpus_hashes,
        'queries_sha256': hashlib.sha256(query_bytes).hexdigest(),
        'python_version': sys.version.split()[0],
        'packages': installed_package_versions() if args.provider == 'local' else {},
        'file_count': len(pages), 'chunk_count': len(docs), 'chunks_per_file': counts,
        'cache_hits': embedder.hits, 'cache_misses': embedder.misses,
        'queries': results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'benchmark.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'baseline.json').write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.output:
        lines = [
            f"Strategy: {type(chunker).__name__} {vars(chunker)}",
            f"Embedding: {args.provider} / {model}",
            f"Corpus: {len(pages)} files -> {len(docs)} chunks",
            f"Top-k: 3",
            "",
        ]
        for q in results:
            lines.extend([
                f"{q['id']}: {q['query']}",
                f"Gold: {q['gold_answer']}",
                f"Filter: {json.dumps(q['metadata_filter'], ensure_ascii=False)}",
            ])
            for rank, hit in enumerate(q['top3'], 1):
                snippet = hit['content'].replace('\n', ' ')[:240]
                lines.append(
                    f"  {rank}. score={hit['score']:.6f} chunk={hit['id']} "
                    f"doc_id={hit['metadata']['doc_id']} audience={hit['metadata']['audience']}"
                )
                lines.append(f"     {snippet}")
            lines.append("Evidence chunks: " + (", ".join(q['evidence_chunk_ids']) or "none"))
            if q['metadata_filter']:
                lines.append("A/B without filter:")
                for rank, hit in enumerate(q['unfiltered_top3'], 1):
                    lines.append(
                        f"  {rank}. score={hit['score']:.6f} chunk={hit['id']} "
                        f"doc_id={hit['metadata']['doc_id']} audience={hit['metadata']['audience']}"
                    )
            lines.append("")
        lines.append(f"Cache: {embedder.hits} hits, {embedder.misses} misses")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f'\nCache hits={embedder.hits}, misses={embedder.misses}; results: {args.output_dir}')
    print('CP5 complete: retrieval only; answer correctness and scoring belong to CP6.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
