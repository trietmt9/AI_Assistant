"""Score retrieval against the golden set, separately from generation.

PLAN.md §10 is emphatic about this and it is the reason this script exists
before the agent does: **retrieval failures and generation failures need
opposite fixes.** A wrong answer because the right chunk was never fetched is a
chunking or embedding problem; a wrong answer from the right chunk is a prompting
or model problem. Collapsing them into one "accuracy" number hides which you
have, and phase 1 is where that distinction is cheapest to preserve.

This measures only the first kind. Generation scoring arrives with the phase-2
agent.

Usage:
    .venv/bin/python eval/retrieval_eval.py              # default config
    .venv/bin/python eval/retrieval_eval.py --compare    # the ablation matrix
    .venv/bin/python eval/retrieval_eval.py --k 10 --verified-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.retrieval.search import Retriever  # noqa: E402
from assistant.retrieval.store import ChunkStore  # noqa: E402

GOLDEN = Path(__file__).parent / "golden.jsonl"


@dataclass(slots=True)
class Question:
    id: str
    question: str
    expect_paths: list[str]
    expect_answer: str = ""
    tier: str = "fast"
    verified: bool = False


@dataclass
class QResult:
    q: Question
    hit_rank: int | None  # 1-indexed rank of the first correct doc, None if missed
    latency_ms: float
    retrieved: list[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None

    @property
    def rr(self) -> float:
        return 1.0 / self.hit_rank if self.hit_rank else 0.0


def load_questions(path: Path = GOLDEN) -> list[Question]:
    out: list[Question] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_comment" in obj and "id" not in obj:
            continue
        out.append(
            Question(
                id=obj["id"],
                question=obj["question"],
                expect_paths=obj.get("expect_paths", []),
                expect_answer=obj.get("expect_answer", ""),
                tier=obj.get("tier", "fast"),
                verified=bool(obj.get("verified", False)),
            )
        )
    return out


def _matches(doc_path: str, expected: list[str]) -> bool:
    return any(e in doc_path for e in expected)


def evaluate(
    questions: list[Question],
    retriever: Retriever,
    *,
    k: int = 5,
    mode: str = "hybrid",
    rerank: bool = True,
    candidates: int = 30,
) -> list[QResult]:
    results: list[QResult] = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(
            q.question, k=k, mode=mode, rerank=rerank, candidates=candidates
        )
        elapsed = (time.perf_counter() - t0) * 1000

        rank = None
        for i, h in enumerate(hits, 1):
            if _matches(h.doc_path, q.expect_paths):
                rank = i
                break
        results.append(
            QResult(
                q=q,
                hit_rank=rank,
                latency_ms=elapsed,
                retrieved=[h.citation for h in hits],
            )
        )
    return results


def summarise(results: list[QResult], k: int) -> dict:
    n = len(results)
    if not n:
        return {"n": 0}
    hits = sum(1 for r in results if r.hit)
    top1 = sum(1 for r in results if r.hit_rank == 1)
    mrr = sum(r.rr for r in results) / n
    lat = sorted(r.latency_ms for r in results)
    return {
        "n": n,
        f"hit@{k}": hits / n,
        "hit@1": top1 / n,
        "mrr": mrr,
        "p50_ms": lat[n // 2],
        "p95_ms": lat[min(n - 1, int(n * 0.95))],
    }


def print_report(results: list[QResult], k: int, label: str = "") -> None:
    s = summarise(results, k)
    head = f"  {label}" if label else ""
    print(f"\n{'=' * 74}")
    print(f"Retrieval eval{head} — {s['n']} questions")
    print("=" * 74)
    print(
        f"  hit@{k}  {s[f'hit@{k}']:.0%}"
        f"     hit@1  {s['hit@1']:.0%}"
        f"     MRR  {s['mrr']:.3f}"
        f"     p50  {s['p50_ms']:.0f}ms"
        f"     p95  {s['p95_ms']:.0f}ms"
    )

    misses = [r for r in results if not r.hit]
    if misses:
        print(f"\n  MISSED ({len(misses)}):")
        for r in misses:
            print(f"    [{r.q.id}] {r.q.question}")
            print(f"        wanted: {', '.join(r.q.expect_paths)}")
            for cite in r.retrieved[:3]:
                print(f"        got:    {cite[:88]}")
    else:
        print("\n  no misses")


def compare(questions: list[Question], retriever: Retriever, k: int) -> None:
    """The ablation that decides the retrieval config.

    PLAN.md §4 asserts hybrid + rerank is right. This is where that assertion
    gets tested on *this* corpus instead of taken on faith -- and where the cost
    of reranking is weighed against what it buys, which matters because the
    cross-encoder is the slowest stage in a voice turn (PLAN.md §5).
    """
    configs = [
        ("vector only", {"mode": "vector", "rerank": False}),
        ("fts only", {"mode": "fts", "rerank": False}),
        ("hybrid, no rerank", {"mode": "hybrid", "rerank": False}),
        ("hybrid + rerank", {"mode": "hybrid", "rerank": True}),
    ]
    rows = []
    for label, kw in configs:
        res = evaluate(questions, retriever, k=k, **kw)
        s = summarise(res, k)
        rows.append((label, s))

    print(f"\n{'=' * 74}")
    print(f"Ablation — {len(questions)} questions, k={k}")
    print("=" * 74)
    print(f"{'config':<22}{f'hit@{k}':>9}{'hit@1':>9}{'MRR':>9}{'p50':>10}{'p95':>10}")
    print("-" * 74)
    for label, s in rows:
        print(
            f"{label:<22}{s[f'hit@{k}']:>8.0%}{s['hit@1']:>9.0%}"
            f"{s['mrr']:>9.3f}{s['p50_ms']:>9.0f}ms{s['p95_ms']:>9.0f}ms"
        )
    print("=" * 74)
    print("Pick on hit@k first, MRR as the tie-break, latency only to reject.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "fts"])
    # Tri-state on purpose. Defaulting to the configured value means a bare run
    # measures what is actually deployed; an earlier version hard-defaulted to
    # reranking on and quietly scored a config nothing uses.
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--compare", action="store_true", help="run the ablation matrix")
    ap.add_argument("--verified-only", action="store_true", help="only human-checked questions")
    ap.add_argument("--tier", action="append", help="restrict to tier(s); repeatable")
    ap.add_argument("--json", type=Path, help="write raw results here")
    args = ap.parse_args()

    store = ChunkStore()
    if store.count() == 0:
        raise SystemExit("index is empty -- run the ingest first (scripts/phase1.md)")

    questions = load_questions()
    if args.verified_only:
        questions = [q for q in questions if q.verified]
    if args.tier:
        questions = [q for q in questions if q.tier in set(args.tier)]
    if not questions:
        raise SystemExit("no questions match those filters")

    unverified = sum(1 for q in questions if not q.verified)
    if unverified:
        print(
            f"note: {unverified}/{len(questions)} questions are unverified stubs. "
            "Their expect_paths are a guess; treat the score as provisional.",
            file=sys.stderr,
        )

    retriever = Retriever()
    print(f"index: {store.count()} chunks")

    if args.compare:
        compare(questions, retriever, args.k)
        return

    from assistant.config import settings

    rerank = settings.rerank_enabled if args.rerank is None else args.rerank
    results = evaluate(
        questions,
        retriever,
        k=args.k,
        mode=args.mode,
        rerank=rerank,
        candidates=args.candidates,
    )
    label = f"{args.mode}{' + rerank' if rerank else ''}"
    print_report(results, args.k, label)

    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "id": r.q.id,
                        "question": r.q.question,
                        "hit_rank": r.hit_rank,
                        "latency_ms": r.latency_ms,
                        "retrieved": r.retrieved,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
