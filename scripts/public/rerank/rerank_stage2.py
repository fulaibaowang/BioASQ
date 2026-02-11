from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    raise ImportError("Missing sentence-transformers. Run: pip install sentence-transformers") from e

from retrieval_eval.common import (
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    normalize_pmid,
    recall_at_k,
    run_df_to_run_map,
)


@dataclass
class OutputConfig:
    output_dir: Path
    runs_dir: Path
    per_query_dir: Path
    adaptive_dir: Path
    metrics_path: Path
    config_path: Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 reranking with a cross-encoder reranker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--runs-dir", type=Path, default=None, help="Directory with stage-1 run TSV files.")
    inputs.add_argument("--run-files", type=Path, nargs="*", default=None, help="Explicit run TSV files.")
    inputs.add_argument("--run-glob", type=str, default="*.tsv", help="Glob for run files under --runs-dir.")
    inputs.add_argument("--docs-jsonl", type=Path, default=None, help="JSONL corpus with PubMed texts.")
    inputs.add_argument("--gold-dir", type=Path, default=None, help="Directory with BioASQ golden JSON files.")
    inputs.add_argument(
        "--gold-files",
        type=str,
        nargs="*",
        default=["13B1_golden.json", "13B2_golden.json", "13B3_golden.json", "13B4_golden.json"],
        help="Golden JSON filenames under --gold-dir.",
    )
    inputs.add_argument(
        "--subset-json",
        type=Path,
        default=None,
        help="Training subset JSON (used when a run name is train_subset).",
    )
    inputs.add_argument(
        "--queries-json",
        type=Path,
        nargs="*",
        default=None,
        help="Extra BioASQ-style JSON files for queries (no gold needed).",
    )
    inputs.add_argument("--candidate-limit", type=int, default=2000, help="Stage-1 candidate cutoff per query.")
    inputs.add_argument("--max-queries", type=int, default=None, help="Max queries per split.")

    model = parser.add_argument_group("model")
    model.add_argument("--model", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    model.add_argument(
        "--model-device",
        type=str,
        default="auto",
        help="Model device: auto, cuda, mps, or cpu.",
    )
    model.add_argument("--model-batch", type=int, default=16, help="Batch size for cross-encoder scoring.")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--use-multi-gpu", action="store_true", help="Enable multi-GPU reranking.")
    runtime.add_argument("--num-gpus", type=int, default=0, help="Max GPUs to use (0 = all).")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--disable-metrics", action="store_true", help="Skip metrics and adaptive cutoff.")
    evaluation.add_argument("--adaptive-p", type=float, default=0.95, help="Target recall ratio for adaptive K.")
    evaluation.add_argument("--adaptive-cap", type=int, default=300, help="Max adaptive cutoff K.")
    evaluation.add_argument(
        "--ks-recall",
        type=str,
        default="50,100,200,300,500,1000,2000",
        help="Recall K values as a comma-separated list.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=None, help="Base output directory.")
    output.add_argument(
        "--category-name",
        type=str,
        default=None,
        help="Optional category name for output subdir and filename prefix.",
    )

    return parser.parse_args()


def _parse_ks_recall(raw: str) -> Tuple[int, ...]:
    if not raw:
        return ()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _resolve_device(model_device: str) -> str:
    if model_device != "auto":
        return model_device
    if torch and torch.cuda.is_available():
        return "cuda"
    if torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid") or cols.get("query_id") or df.columns[0]
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc") or df.columns[1]
    rank_col = cols.get("rank")
    score_col = cols.get("score")

    out = pd.DataFrame(
        {
            "qid": df[qid_col].astype(str),
            "docno": df[doc_col].astype(str).map(normalize_pmid),
        }
    )

    if rank_col:
        out["rank"] = df[rank_col].astype(int)
    else:
        out["rank"] = out.groupby("qid").cumcount() + 1

    if score_col:
        out["score"] = df[score_col].astype(float)
    else:
        out["score"] = np.nan

    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


def extract_docno(rec: dict) -> str:
    for key in ("docno", "pmid", "id"):
        if key in rec:
            return normalize_pmid(rec[key])
    return ""


def extract_text(rec: dict) -> str:
    if "text" in rec and rec["text"]:
        return str(rec["text"]).strip()
    parts = []
    if rec.get("title"):
        parts.append(str(rec["title"]).strip())
    if rec.get("abstract"):
        parts.append(str(rec["abstract"]).strip())
    if rec.get("abstractText"):
        parts.append(str(rec["abstractText"]).strip())
    return " ".join([p for p in parts if p])


def load_doc_texts(docnos: Iterable[str], jsonl_path: Path) -> Dict[str, str]:
    wanted = set(map(str, docnos))
    out: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            docno = extract_docno(rec)
            if docno in wanted and docno not in out:
                out[docno] = extract_text(rec)
                if len(out) == len(wanted):
                    break
    return out


def _chunk_items(items: List[Tuple[str, List[str]]], n: int) -> List[List[Tuple[str, List[str]]]]:
    if n <= 1:
        return [items]
    chunk_size = max(1, math.ceil(len(items) / n))
    return [
        items[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n)
        if items[i * chunk_size : (i + 1) * chunk_size]
    ]


def _rerank_worker(
    gpu_id: int,
    items: List[Tuple[str, List[str]]],
    topics: Dict[str, str],
    doc_texts: Dict[str, str],
    model_name: str,
    batch_size: int,
    return_dict,
) -> None:
    device = f"cuda:{gpu_id}" if torch and torch.cuda.is_available() else "cpu"
    model = CrossEncoder(model_name, device=device)
    local_out: Dict[str, List[Tuple[str, float]]] = {}
    total = len(items)
    start = time()
    for idx, (qid, docs) in enumerate(items, start=1):
        query = topics.get(qid, "").strip()
        if not query:
            local_out[qid] = [(doc, float("nan")) for doc in docs]
            continue
        pairs = [(query, doc_texts.get(doc, "")) for doc in docs]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored = [(doc, float(score)) for doc, score in zip(docs, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        local_out[qid] = scored
        if idx == 1 or idx % 10 == 0 or idx == total:
            elapsed = max(1e-9, time() - start)
            rate = idx / elapsed
            print(f"[gpu {gpu_id}] {idx}/{total} queries | {rate:.2f} q/s")
    return_dict[gpu_id] = local_out


def rerank_run(
    run_map: Dict[str, List[str]],
    topics: Dict[str, str],
    doc_texts: Dict[str, str],
    model: CrossEncoder | None,
    model_name: str,
    batch_size: int,
    use_multi_gpu: bool,
    num_gpus: int,
) -> Dict[str, List[Tuple[str, float]]]:
    items = list(run_map.items())
    if use_multi_gpu:
        if not torch or not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU requested but CUDA is not available.")
        available = torch.cuda.device_count()
        if available < 2:
            raise RuntimeError("Multi-GPU requested but fewer than 2 CUDA devices found.")
        use_n = available if not num_gpus or num_gpus < 1 else min(num_gpus, available)
        chunks = _chunk_items(items, use_n)
        ctx = torch.multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        procs = []
        for gpu_id, chunk in enumerate(chunks):
            p = ctx.Process(
                target=_rerank_worker,
                args=(gpu_id, chunk, topics, doc_texts, model_name, batch_size, return_dict),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        merged: Dict[str, List[Tuple[str, float]]] = {}
        for part in return_dict.values():
            merged.update(part)
        return {qid: merged.get(qid, [(doc, float("nan")) for doc in docs]) for qid, docs in items}

    if model is None:
        raise RuntimeError("Single-GPU path requires a loaded model instance.")

    out: Dict[str, List[Tuple[str, float]]] = {}
    total = len(items)
    start = time()
    for idx, (qid, docs) in enumerate(items, start=1):
        query = topics.get(qid, "").strip()
        if not query:
            out[qid] = [(doc, float("nan")) for doc in docs]
            continue
        pairs = [(query, doc_texts.get(doc, "")) for doc in docs]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored = [(doc, float(score)) for doc, score in zip(docs, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        out[qid] = scored
        if idx == 1 or idx % 10 == 0 or idx == total:
            elapsed = max(1e-9, time() - start)
            rate = idx / elapsed
            print(f"[rerank] {idx}/{total} queries | {rate:.2f} q/s")
    return out


def adaptive_k_for_query(gold: List[str], ranked: List[str], p: float, cap: int) -> int:
    gold_set = set(map(str, gold))
    if not gold_set:
        return 0
    max_recall = recall_at_k(gold_set, ranked, k=len(ranked))
    target = p * max_recall
    if target <= 0:
        return 0
    k_max = min(cap, len(ranked))
    for k in range(1, k_max + 1):
        if recall_at_k(gold_set, ranked, k=k) >= target:
            return k
    return k_max


def evaluate_adaptive(
    gold_map: Dict[str, List[str]],
    run_map: Dict[str, List[str]],
    p: float,
    cap: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    recalls = []
    keffs = []
    shortfalls = []

    for qid, gold in gold_map.items():
        ranked = run_map.get(qid, [])
        k_rec = adaptive_k_for_query(gold, ranked, p=p, cap=cap)
        k_eff = min(k_rec, len(ranked))
        shortfall = 1.0 if k_rec > len(ranked) else 0.0
        r = recall_at_k(set(map(str, gold)), ranked, k=k_eff) if k_eff > 0 else 0.0

        rows.append({"qid": qid, "K_rec": k_rec, "K_eff": k_eff, "R@Krec": r})
        recalls.append(r)
        keffs.append(k_eff)
        shortfalls.append(shortfall)

    metrics = {
        "MeanR@Krec": float(np.mean(recalls)) if recalls else 0.0,
        "MeanKeff@Krec": float(np.mean(keffs)) if keffs else 0.0,
        "ShortfallRate@Krec": float(np.mean(shortfalls)) if shortfalls else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def _build_output_config(base_dir: Path, category_name: str | None) -> OutputConfig:
    output_dir = base_dir
    if category_name:
        output_dir = output_dir / category_name
    runs_dir = output_dir / "runs"
    per_query_dir = output_dir / "per_query"
    adaptive_dir = output_dir / "adaptive"

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir.mkdir(parents=True, exist_ok=True)
    adaptive_dir.mkdir(parents=True, exist_ok=True)

    return OutputConfig(
        output_dir=output_dir,
        runs_dir=runs_dir,
        per_query_dir=per_query_dir,
        adaptive_dir=adaptive_dir,
        metrics_path=output_dir / "metrics.csv",
        config_path=output_dir / "config.json",
    )


def _category_prefix(category_name: str | None) -> str:
    return f"{category_name}__" if category_name else ""


def main() -> None:
    args = parse_args()
    root = _resolve_repo_root()

    runs_dir = args.runs_dir or root / "output" / "eval_hybird_production_test" / "runs"
    docs_jsonl = args.docs_jsonl or root / "output" / "subset_pubmed.jsonl"
    gold_dir = args.gold_dir or root / "bioasq_data" / "Task13BGoldenEnriched"
    subset_json = args.subset_json or root / "example" / "training14b_10pct_sample.json"
    output_dir = args.output_dir or root / "output" / "eval_stage2_rerank"

    output_cfg = _build_output_config(output_dir, args.category_name)
    prefix = _category_prefix(args.category_name)

    if args.run_files:
        run_files = [Path(p) for p in args.run_files]
    else:
        run_files = sorted(runs_dir.glob(args.run_glob))

    if not run_files:
        raise FileNotFoundError("No run files found. Provide --run-files or --runs-dir/--run-glob.")

    ks_recall = _parse_ks_recall(args.ks_recall)

    run_dfs: Dict[str, pd.DataFrame] = {}
    for path in run_files:
        name = path.stem
        df = load_run_tsv(path)
        if args.candidate_limit:
            df = df[df["rank"] <= int(args.candidate_limit)]
        if args.max_queries:
            qids = sorted(df["qid"].unique())[: int(args.max_queries)]
            df = df[df["qid"].isin(qids)]
        run_dfs[name] = df

    run_maps = {name: run_df_to_run_map(df) for name, df in run_dfs.items()}
    run_names = list(run_maps.keys())

    candidate_docnos = set()
    for docs in run_maps.values():
        for doc_list in docs.values():
            candidate_docnos.update(doc_list)

    print("candidate docnos:", len(candidate_docnos))
    doc_texts = load_doc_texts(candidate_docnos, docs_jsonl)
    print("loaded texts:", len(doc_texts))

    topics_map: Dict[str, str] = {}
    gold_maps: Dict[str, Dict[str, List[str]]] = {}

    if gold_dir.exists():
        for filename in args.gold_files:
            gold_path = gold_dir / filename
            if not gold_path.exists():
                continue
            questions = load_questions(gold_path)
            topics_df, gold_map = build_topics_and_gold(questions)
            topics_map.update(dict(zip(topics_df["qid"], topics_df["query"])))
            gold_maps[str(gold_path.stem)] = gold_map

    if subset_json.exists() and "train_subset" in run_names:
        questions = load_questions(subset_json)
        topics_df, gold_map = build_topics_and_gold(questions)
        topics_map.update(dict(zip(topics_df["qid"], topics_df["query"])))
        gold_maps["train_subset"] = gold_map

    if args.queries_json:
        for qpath in args.queries_json:
            qpath = Path(qpath)
            if not qpath.exists():
                continue
            questions = load_questions(qpath)
            topics_df, _ = build_topics_and_gold(questions)
            topics_map.update(dict(zip(topics_df["qid"], topics_df["query"])))

    model_device = _resolve_device(args.model_device)

    reranker = None
    if not args.use_multi_gpu:
        reranker = CrossEncoder(args.model, device=model_device)

    reranked_runs: Dict[str, List[Tuple[str, float]]] = {}
    for name in run_names:
        reranked = rerank_run(
            run_maps[name],
            topics=topics_map,
            doc_texts=doc_texts,
            model=reranker,
            model_name=args.model,
            batch_size=args.model_batch,
            use_multi_gpu=args.use_multi_gpu,
            num_gpus=args.num_gpus,
        )
        reranked_runs[name] = reranked
        print("reranked", name, "queries:", len(reranked))

    for name, reranked in reranked_runs.items():
        rows = []
        for qid, docs in reranked.items():
            for rank, (docno, score) in enumerate(docs, start=1):
                rows.append({"qid": qid, "docno": docno, "rank": rank, "score": score})
        run_df = pd.DataFrame(rows)
        out_path = output_cfg.runs_dir / f"{prefix}{name}.tsv"
        run_df.to_csv(out_path, sep="\t", index=False)

    adaptive_stats = {}
    summary_rows = []

    if not args.disable_metrics:
        for name, reranked in reranked_runs.items():
            if name not in gold_maps:
                print("skip metrics, no gold for", name)
                continue
            reranked_map = {qid: [doc for doc, _ in docs] for qid, docs in reranked.items()}
            metrics, perq = evaluate_run(gold_maps[name], reranked_map, ks_recall=ks_recall)
            perq.to_csv(output_cfg.per_query_dir / f"{prefix}{name}.csv", index=False)

            adaptive_metrics, adaptive_perq = evaluate_adaptive(
                gold_maps[name], reranked_map, p=args.adaptive_p, cap=args.adaptive_cap
            )
            adaptive_perq.to_csv(output_cfg.adaptive_dir / f"{prefix}{name}.csv", index=False)
            adaptive_stats[name] = adaptive_metrics

            row = {"split": name}
            row.update(metrics)
            row.update(adaptive_metrics)
            summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(output_cfg.metrics_path, index=False)

    config = {
        "model": args.model,
        "model_device": model_device,
        "model_batch": args.model_batch,
        "use_multi_gpu": args.use_multi_gpu,
        "num_gpus": args.num_gpus,
        "candidate_limit": args.candidate_limit,
        "max_queries": args.max_queries,
        "runs_dir": str(runs_dir),
        "run_files": [str(p) for p in run_files],
        "docs_jsonl": str(docs_jsonl),
        "gold_dir": str(gold_dir),
        "gold_files": args.gold_files,
        "subset_json": str(subset_json),
        "queries_json": [str(p) for p in args.queries_json or []],
        "ks_recall": list(ks_recall),
        "adaptive_p": args.adaptive_p,
        "adaptive_cap": args.adaptive_cap,
        "category_name": args.category_name,
    }
    output_cfg.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("saved outputs to", output_cfg.output_dir)


if __name__ == "__main__":
    main()
