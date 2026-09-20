#!/usr/bin/env python3
"""Score CP6 retrieval at document and evidence-chunk levels.

This intentionally does not judge a generated answer. The default project LLM
is a prompt-preview stub, so claiming agent correctness would be fabricated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evidence_rank(query: dict) -> int | None:
    for rank, hit in enumerate(query['top3'], start=1):
        if query['evidence_quote'] in hit['content']:
            return rank
    return None


def document_rank(query: dict) -> int | None:
    for rank, hit in enumerate(query['top3'], start=1):
        if hit['metadata']['doc_id'] == query['source_doc_id']:
            return rank
    return None


def run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = []
    for query in data['queries']:
        doc_rank = document_rank(query)
        quote_rank = evidence_rank(query)
        # Retrieval-only proxy follows the course rank convention, but does
        # not add the required agent-answer point.
        retrieval_points = 0 if quote_rank is None else 2 if quote_rank == 1 else 1
        rows.append({
            'id': query['id'], 'gold_doc_id': query['source_doc_id'],
            'document_rank': doc_rank, 'evidence_rank': quote_rank,
            'document_in_top3': doc_rank is not None,
            'evidence_in_top3': quote_rank is not None,
            'retrieval_proxy_points': retrieval_points,
            'top3': [{'id': h['id'], 'doc_id': h['metadata']['doc_id'],
                      'audience': h['metadata']['audience'], 'score': h['score']}
                     for h in query['top3']],
            'filter': query['metadata_filter'],
            'unfiltered_top3': [{'id': h['id'], 'doc_id': h['metadata']['doc_id'],
                                 'audience': h['metadata']['audience'], 'score': h['score']}
                                for h in query['unfiltered_top3']],
        })
    return {
        'strategy': data['strategy'], 'strategy_parameters': data['strategy_parameters'],
        'embedding_provider': data['embedding_provider'],
        'embedding_model': data['embedding_model'], 'chunk_count': data['chunk_count'],
        'rows': rows,
        'document_level_hits': sum(row['document_in_top3'] for row in rows),
        'evidence_level_hits': sum(row['evidence_in_top3'] for row in rows),
        'retrieval_proxy_points': sum(row['retrieval_proxy_points'] for row in rows),
        'agent_answers_evaluated': False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    outputs = {path.parent.name: run(path) for path in args.benchmark}
    args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, result in outputs.items():
        print(f"{name}: chunks={result['chunk_count']}; document={result['document_level_hits']}/5; "
              f"evidence={result['evidence_level_hits']}/5; "
              f"retrieval-proxy={result['retrieval_proxy_points']}/10; agent=not evaluated")
        for row in result['rows']:
            print(f"  {row['id']}: doc rank={row['document_rank']}; evidence rank={row['evidence_rank']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
