# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     jupytext_version: 1.19.1
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM `evidence_ids` vs MAP@10 (snippet evidence baseline)
#
# One-off analysis on **legacy wrapped JSON** under a workflow output directory: compare BioASQ AP@10
# for the original `documents` order vs a list reordered by LLM `evidence_ids` (cited PMIDs first, then
# the rest in original order). Pairs of subdirs are configurable; by default we run:
#
# - **default** — `evidence_snippet` + `generation_snippet` (e.g. ~30-doc pool: tail can fill rank 1–10).
# - **llama_10ctx** — `evidence_snippet_` + `generation_snippet_llama_` (~10-doc pool: mostly **reordering** within top 10).
#
# No pipeline code changes.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# %% [markdown]
# ## 1) Paths and config

# %%
try:
    _NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:
    _NOTEBOOK_DIR = Path.cwd()
REPO_ROOT = _NOTEBOOK_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "public" / "shared_scripts"))

from retrieval_eval.common import ap_at_k, normalize_pmid

# Root that contains evidence / generation subdirs (see ARTIFACT_PROFILES below).
WORKFLOW_OUTPUT = REPO_ROOT / "output" / "workflow_baseline_full_run_both_routes_gemma"

# Each profile: logical name, evidence subdir, generation subdir (under WORKFLOW_OUTPUT).
ARTIFACT_PROFILES: List[Dict[str, str]] = [
    {"name": "snippet_default", "evidence_dir": "evidence_snippet", "generation_dir": "generation_snippet"},
    {
        "name": "llama_10ctx",
        "evidence_dir": "evidence_snippet_",
        "generation_dir": "generation_snippet_llama_",
    },
]

# Default gold: Task 13B golden JSON (wrapped). Optional overrides for extra stems (e.g. training subset).
GOLD_DIR = REPO_ROOT / "bioasq_data" / "Task13BGoldenEnriched"
# If a nonstandard stem appears (e.g. training sample), map stem -> golden JSON path.
GOLD_OVERRIDES: Dict[str, Path] = {
    "training14b_10pct_sample": REPO_ROOT / "example" / "training14b_10pct_sample.json",
}

EPS = 1e-9

# %% [markdown]
# ## 2) Load wrapped BioASQ JSON (`{"questions": [...]}`)

# %%


def load_wrapped_questions(path: Path) -> List[dict]:
    """Single-file wrapped BioASQ JSON (not JSONL)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"{path}: expected top-level 'questions' list")
    return questions


def documents_to_pmids_ordered(documents: Any) -> List[str]:
    """Normalize documents URLs/PMIDs to PMIDs; dedupe preserving first occurrence."""
    if not isinstance(documents, list):
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for d in documents:
        p = normalize_pmid(d)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def evidence_ids_to_cited_pmids(evidence_ids: Any) -> Set[str]:
    """Parse passage ids like `39555889-1` -> PMID `39555889`."""
    out: Set[str] = set()
    if not isinstance(evidence_ids, list):
        return out
    for raw in evidence_ids:
        s = str(raw).strip()
        if not s:
            continue
        head = s.split("-", 1)[0].strip()
        p = normalize_pmid(head)
        if p:
            out.add(p)
    return out


def reorder_docs_with_llm(original_pmids: List[str], evidence_ids: Any) -> List[str]:
    """
    Cited PMIDs (from evidence_ids) that appear in `original_pmids` first, in **original retrieval order**;
    then append the rest of `original_pmids` in order. Same length as deduped original when every cited
    PMID is in the pool.
    """
    if not original_pmids:
        return []
    cited = evidence_ids_to_cited_pmids(evidence_ids)
    head = [p for p in original_pmids if p in cited]
    head_set = set(head)
    tail = [p for p in original_pmids if p not in head_set]
    return head + tail


def build_gold_map_from_wrapped(path: Path) -> Dict[str, List[str]]:
    """qid -> list of relevant PMIDs (order preserved; duplicates stripped)."""
    questions = load_wrapped_questions(path)
    gold: Dict[str, List[str]] = {}
    for q in questions:
        qid = str(q.get("id", "")).strip()
        if not qid:
            continue
        pmids = documents_to_pmids_ordered(q.get("documents"))
        gold[qid] = pmids
    return gold


def stem_from_contexts_filename(name: str) -> str | None:
    """`13B1_golden_contexts.json` -> `13B1_golden`."""
    if not name.endswith("_contexts.json"):
        return None
    return name[: -len("_contexts.json")]


def resolve_gold_path(stem: str) -> Path | None:
    if stem in GOLD_OVERRIDES:
        p = GOLD_OVERRIDES[stem]
        return p if p.is_file() else None
    p = GOLD_DIR / f"{stem}.json"
    return p if p.is_file() else None


def discover_splits(workflow_out: Path, evidence_dir: str, generation_dir: str) -> List[str]:
    ev_dir = workflow_out / evidence_dir
    gen_dir = workflow_out / generation_dir
    if not ev_dir.is_dir() or not gen_dir.is_dir():
        print(f"[skip] missing dir: {ev_dir} or {gen_dir}")
        return []
    stems: List[str] = []
    for p in sorted(ev_dir.glob("*_contexts.json")):
        stem = stem_from_contexts_filename(p.name)
        if not stem:
            continue
        ans = gen_dir / f"{stem}_answers.json"
        if ans.is_file():
            stems.append(stem)
        else:
            print(f"[skip] no generation file for stem {stem!r}: {ans}")
    return stems

# %% [markdown]
# ## 3) Per-split evaluation

# %%


def evaluate_split(
    stem: str,
    workflow_out: Path,
    gold_map: Dict[str, List[str]],
    evidence_dir: str,
    generation_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ev_path = workflow_out / evidence_dir / f"{stem}_contexts.json"
    gen_path = workflow_out / generation_dir / f"{stem}_answers.json"

    ev_questions = {str(q.get("id")): q for q in load_wrapped_questions(ev_path) if q.get("id") is not None}
    gen_questions = {str(q.get("id")): q for q in load_wrapped_questions(gen_path) if q.get("id") is not None}

    rows: List[dict] = []
    qids = sorted(set(ev_questions.keys()) & set(gen_questions.keys()), key=lambda x: x)
    missing_gen = set(ev_questions.keys()) - set(gen_questions.keys())
    missing_ev = set(gen_questions.keys()) - set(ev_questions.keys())
    if missing_gen:
        print(f"  [{stem}] evidence rows without generation: {len(missing_gen)}")
    if missing_ev:
        print(f"  [{stem}] generation rows without evidence: {len(missing_ev)}")

    for qid in qids:
        ev = ev_questions[qid]
        gen = gen_questions[qid]
        orig = documents_to_pmids_ordered(ev.get("documents"))
        reord = reorder_docs_with_llm(orig, gen.get("evidence_ids"))
        gold_list = gold_map.get(qid, [])
        gold_set = set(gold_list)

        ap_o = ap_at_k(gold_set, orig, k=10)
        ap_r = ap_at_k(gold_set, reord, k=10)
        delta = ap_r - ap_o
        if delta > EPS:
            bucket = "improved"
        elif delta < -EPS:
            bucket = "worse"
        else:
            bucket = "unchanged"

        n_cited_in_pool = len(evidence_ids_to_cited_pmids(gen.get("evidence_ids")) & set(orig))

        rows.append(
            {
                "qid": qid,
                "AP@10_orig": ap_o,
                "AP@10_llm": ap_r,
                "delta": delta,
                "bucket": bucket,
                "n_docs_pool": len(orig),
                "n_evidence_ids": len(gen.get("evidence_ids") or []),
                "n_cited_in_pool": n_cited_in_pool,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "MAP@10_orig": float("nan"),
            "MAP@10_llm": float("nan"),
            "mean_delta": float("nan"),
            "n_improved": 0,
            "n_unchanged": 0,
            "n_worse": 0,
            "n_queries": 0,
        }
        return df, summary

    summary = {
        "MAP@10_orig": float(df["AP@10_orig"].mean()),
        "MAP@10_llm": float(df["AP@10_llm"].mean()),
        "mean_delta": float(df["delta"].mean()),
        "n_improved": int((df["bucket"] == "improved").sum()),
        "n_unchanged": int((df["bucket"] == "unchanged").sum()),
        "n_worse": int((df["bucket"] == "worse").sum()),
        "n_queries": len(df),
    }
    return df, summary


def print_example_row(
    stem: str,
    workflow_out: Path,
    gold_map: Dict[str, List[str]],
    evidence_dir: str,
    generation_dir: str,
    qid: str | None = None,
) -> None:
    ev_path = workflow_out / evidence_dir / f"{stem}_contexts.json"
    gen_path = workflow_out / generation_dir / f"{stem}_answers.json"
    ev_questions = {str(q.get("id")): q for q in load_wrapped_questions(ev_path) if q.get("id") is not None}
    gen_questions = {str(q.get("id")): q for q in load_wrapped_questions(gen_path) if q.get("id") is not None}
    common = sorted(set(ev_questions.keys()) & set(gen_questions.keys()))
    if not common:
        print("No overlapping qids for example.")
        return
    pick = qid if qid and qid in common else common[0]
    ev = ev_questions[pick]
    gen = gen_questions[pick]
    orig = documents_to_pmids_ordered(ev.get("documents"))
    reord = reorder_docs_with_llm(orig, gen.get("evidence_ids"))
    gold_list = gold_map.get(pick, [])
    print(f"--- Example qid={pick} ({stem}) ---")
    print("gold PMIDs:", gold_list[:20], "..." if len(gold_list) > 20 else "")
    print("evidence_ids:", gen.get("evidence_ids"))
    print("original top-10:", orig[:10])
    print("reordered top-10:", reord[:10])

# %% [markdown]
# ## 4) Run all splits (each artifact profile)

# %%
print("Workflow:", WORKFLOW_OUTPUT)

all_summaries: List[Dict[str, Any]] = []
per_split_frames: Dict[str, Dict[str, pd.DataFrame]] = {}

for profile in ARTIFACT_PROFILES:
    pname = profile["name"]
    ev_sub = profile["evidence_dir"]
    gen_sub = profile["generation_dir"]
    per_split_frames[pname] = {}

    print(f"\n######## Profile: {pname}  ({ev_sub!r} + {gen_sub!r}) ########")
    stems = discover_splits(WORKFLOW_OUTPUT, ev_sub, gen_sub)
    print("Discovered stems:", stems)
    if not stems:
        continue

    for stem in stems:
        gold_path = resolve_gold_path(stem)
        if gold_path is None:
            print(f"[skip] no gold file for stem {stem!r} (checked GOLD_DIR / {stem}.json and GOLD_OVERRIDES)")
            continue
        print(f"\n=== [{pname}] Split {stem} | gold: {gold_path} ===")
        gmap = build_gold_map_from_wrapped(gold_path)
        df, summary = evaluate_split(stem, WORKFLOW_OUTPUT, gmap, ev_sub, gen_sub)
        summary["stem"] = stem
        summary["profile"] = pname
        summary["evidence_dir"] = ev_sub
        summary["generation_dir"] = gen_sub
        all_summaries.append(summary)
        per_split_frames[pname][stem] = df
        print(
            f"  MAP@10 orig={summary['MAP@10_orig']:.4f}  llm={summary['MAP@10_llm']:.4f}  "
            f"Δmean={summary['mean_delta']:+.4f}"
        )
        print(
            f"  queries={summary['n_queries']}  improved={summary['n_improved']}  "
            f"unchanged={summary['n_unchanged']}  worse={summary['n_worse']}"
        )
        if not df.empty:
            print_example_row(stem, WORKFLOW_OUTPUT, gmap, ev_sub, gen_sub, qid=str(df.iloc[0]["qid"]))

# %%
summary_df = pd.DataFrame(all_summaries)
summary_df

# %%
if not summary_df.empty:
    for pname in summary_df["profile"].unique():
        sub = summary_df[summary_df["profile"] == pname]
        total_q = sub["n_queries"].sum()
        if total_q <= 0:
            continue
        w_map_orig = (sub["MAP@10_orig"] * sub["n_queries"]).sum() / total_q
        w_map_llm = (sub["MAP@10_llm"] * sub["n_queries"]).sum() / total_q
        print(
            f"[{pname}] Query-weighted MAP@10 across splits (n={int(total_q)}): "
            f"orig={w_map_orig:.4f}  llm={w_map_llm:.4f}"
        )

# %% [markdown]
# ### Per-query tables (optional export)

# %%
# Example: inspect one split
# per_split_frames["snippet_default"]["13B1_golden"].head(20)
# per_split_frames["llama_10ctx"]["13B1_golden"].head(20)
