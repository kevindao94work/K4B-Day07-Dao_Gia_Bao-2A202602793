#!/usr/bin/env python3
"""Validate the CP2 corpus, manifest and grounded benchmark without dependencies.

The restricted front matter emitted by fetch_public_pages.py uses JSON-quoted
strings (a subset of YAML); do not strip quotes with the handout's regex alone.
"""
import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data/tiki-doi-tra'
REQ = {'doc_id', 'title', 'source_url', 'retrieved_at', 'document_version', 'audience'}


def main():
    docs = {}
    for p in sorted(D.glob('*.md')):
        text = p.read_text(encoding='utf-8')
        parts = text.split('---', 2)
        assert len(parts) == 3 and not parts[0].strip(), f'{p}: front matter'
        fm = {k: json.loads(v) for k, v in re.findall(r'^(\w+):\s*(.+)$', parts[1], re.M)}
        assert REQ <= fm.keys() and all(fm[k] for k in REQ), f'{p}: metadata'
        assert fm['doc_id'] == p.stem and p.stem not in docs, f'{p}: doc_id'
        assert re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', p.stem), f'{p}: slug'
        assert fm['audience'] in {'buyer', 'seller', 'both'}, f'{p}: audience'
        assert fm.get('category') and fm.get('channel') and fm.get('platform'), f'{p}: filter fields'
        assert fm['language'] == 'vi' and fm['market'] == 'VN' and fm['platform'] == 'tiki', f'{p}: locale'
        assert fm['source_url'].startswith('https://'), f'{p}: source'
        date.fromisoformat(fm['retrieved_at'])
        assert '\n## ' in parts[2], f'{p}: headings'
        assert not any(x in parts[2] for x in ['Cookies on Tiki', 'Bạn có thấy bài viết hữu ích', 'Sản phẩm liên quan']), f'{p}: noise'
        docs[p.stem] = (fm, parts[2])
        print(f'{p.name:45} OK')
    assert 5 <= len(docs) <= 10, 'document count'
    auds = Counter(m['audience'] for m, body in docs.values())
    assert len(auds) >= 2, 'audience diversity'
    with (D / 'sources.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert sorted(r['doc_id'] for r in rows) == sorted(docs), 'manifest 1:1'
    for r in rows:
        m, _ = docs[r['doc_id']]
        for k in ['doc_id', 'title', 'source_url', 'retrieved_at', 'document_version']:
            assert r[k] == m[k], f'manifest mismatch: {k}'
        assert ROOT / r['file_path'].replace('\\', '/') == D / (r['doc_id'] + '.md'), 'manifest path'
        assert r['license_or_permission'], 'permission basis'
    with (ROOT / 'data/urls.csv').open(encoding='utf-8', newline='') as f:
        inputs = list(csv.DictReader(f))
    assert 5 <= len(inputs) <= 10, '5-10 source URLs'
    assert {r['url'] for r in inputs} == {m['source_url'] for m, _ in docs.values()}, 'URL coverage'
    assert all(r['license_or_permission'] == 'public-source' for r in rows), 'license'
    queries = json.loads((ROOT / 'data/benchmark_queries.json').read_text(encoding='utf-8'))
    assert len(queries) == 5 and len({q['id'] for q in queries}) == 5, '5 queries'
    assert any(q['metadata_filter'].get('audience') in {'buyer', 'seller'} for q in queries), 'audience query'
    for q in queries:
        m, body = docs[q['source_doc_id']]
        assert q['evidence_quote'] in body, f"{q['id']}: evidence not found"
        assert all(m.get(k) == v for k, v in q['metadata_filter'].items()), f"{q['id']}: filter excludes gold"
        assert q['gold_answer'] and q['source_heading'], 'gold answer'
        print(f"{q['id']}: evidence + metadata_filter OK")
    print(f'so file : {len(docs)} OK (can 5-10)')
    print('csv     : khop 1-1 OK')
    print(f'audience: {dict(auds)} OK')
    print('CP2 structural checks: OK; source access and semantic review: see collection log.')


if __name__ == '__main__':
    main()
