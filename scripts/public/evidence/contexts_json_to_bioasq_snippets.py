#!/usr/bin/env python3
"""
Convert pipeline *_contexts.jsonl (or legacy *_contexts.json) to BioASQ-style question JSON (documents + snippets).

- Keeps the first 10 document URLs per question (rerank / post-rerank order).
- For each kept document with a matching context entry:
  - If selected_windows is non-empty: pick the window with highest ce_score (tie: lower
    query_field lexicographically, then lower window_idx), emit one abstract snippet
    (abstract text only; offsets in corpus abstract).
  - If selected_windows is empty: emit one title snippet (offsets in corpus title).

Requires PubMed JSONL corpus (pmid, title, abstract) for correct spans, aligned with
snippet context building (NLTK sentence splits on raw abstract).
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

PUBMED_URL_PATTERN = re.compile(r"pubmed/(\d+)/?$", re.I)
PUBMED_DOCUMENT_PREFIX = "http://www.ncbi.nlm.nih.gov/pubmed/"

_SHARED_SCRIPTS = Path(__file__).resolve().parent.parent / "shared_scripts"
if str(_SHARED_SCRIPTS).is_dir() and str(_SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SHARED_SCRIPTS))
from retrieval_eval.doc_id_util import ranked_doc_ids_from_question  # noqa: E402
_QF_TIE_SENTINEL = "\uffff"
DOC_CAP = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--contexts-jsonl",
        "--contexts-json",
        type=Path,
        required=True,
        dest="contexts_path",
        help="Path to contexts .jsonl (one question dict per line) or legacy wrapped .json with a questions array.",
    )
    p.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path or glob to PubMed JSONL (pmid, title, abstract).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output BioASQ-shaped JSON path.",
    )
    p.add_argument(
        "--allow-fallback-offsets",
        action="store_true",
        help="If abstract alignment fails, emit snippet with offsets 0 and len(text) (abstract section).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def pmid_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    m = PUBMED_URL_PATTERN.search(url.strip())
    return m.group(1) if m else None


def _normalize_unicode_whitespace(text: str) -> str:
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Zl", "Zp") or (cat == "Zs" and ch != " "):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"  +", " ", "".join(out))


def _resolve_corpus_paths(path_or_glob: str) -> List[Path]:
    if "*" in path_or_glob or "?" in path_or_glob:
        paths = sorted(Path(p) for p in glob_mod.glob(path_or_glob) if Path(p).is_file())
        if not paths:
            raise FileNotFoundError(f"No files matched corpus glob: {path_or_glob}")
        return paths
    p = Path(path_or_glob)
    if not p.exists():
        raise FileNotFoundError(f"Corpus not found: {path_or_glob}")
    return [p]


def build_pmid_to_title_abstract(
    corpus_path: str,
    needed_pmids: Set[str],
) -> Dict[str, Tuple[str, str]]:
    """pmid -> (raw title, raw abstract) strings from JSONL."""
    paths = _resolve_corpus_paths(corpus_path)
    out: Dict[str, Tuple[str, str]] = {}
    for fp in paths:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pmid_raw = obj.get("pmid")
                if pmid_raw is None:
                    continue
                pmid = str(pmid_raw).strip()
                if pmid not in needed_pmids or pmid in out:
                    continue
                title = obj.get("title") or ""
                abstract = obj.get("abstract") or obj.get("abstractText") or ""
                if isinstance(title, list):
                    title = " ".join(str(t) for t in title)
                if isinstance(abstract, list):
                    abstract = " ".join(str(a) for a in abstract)
                out[pmid] = (str(title), str(abstract))
                if len(out) == len(needed_pmids):
                    return out
        if len(out) == len(needed_pmids):
            break
    return out


def _ensure_nltk_punkt() -> None:
    import nltk

    try:
        nltk.sent_tokenize("Hello.")
    except LookupError:
        for res in ("punkt_tab", "punkt"):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass


def abstract_sentences(abstract: str) -> List[str]:
    _ensure_nltk_punkt()
    import nltk

    a = abstract.strip()
    if not a:
        return []
    return [s.strip() for s in nltk.sent_tokenize(a) if s.strip()]


def _better_window_tie(
    score: float,
    qf: Optional[str],
    wi: int,
    best_score: float,
    best_qf: Optional[str],
    best_wi: int,
) -> bool:
    """Return True if (score, qf, wi) beats (best_score, best_qf, best_wi). Mirrors CE max-pool tie."""
    if score > best_score:
        return True
    if score < best_score:
        return False
    a = str(qf) if qf is not None else _QF_TIE_SENTINEL
    b = str(best_qf) if best_qf is not None else _QF_TIE_SENTINEL
    if a < b:
        return True
    if a > b:
        return False
    return wi < best_wi


def pick_best_window(selected_windows: List[dict]) -> Optional[dict]:
    best: Optional[dict] = None
    best_score = -1.0
    best_qf: Optional[str] = None
    best_wi = 10**9
    for w in selected_windows:
        try:
            sc = float(w.get("ce_score", 0.0))
        except (TypeError, ValueError):
            sc = 0.0
        qf_raw = w.get("query_field")
        qf: Optional[str] = str(qf_raw) if qf_raw is not None and qf_raw != "" else None
        try:
            wi = int(w.get("window_idx", 0))
        except (TypeError, ValueError):
            wi = 0
        if best is None or _better_window_tie(sc, qf, wi, best_score, best_qf, best_wi):
            best = w
            best_score = sc
            best_qf = qf
            best_wi = wi
    return best


def span_for_sentence_range(
    abstract: str,
    sentences: List[str],
    sent_ids: List[int],
) -> Optional[Tuple[int, int, str]]:
    """Character [start, end) in abstract covering sent_ids; snippet text = abstract[start:end]."""
    ids = sorted({int(i) for i in sent_ids if i is not None})
    if not ids:
        return None
    lo, hi = ids[0], ids[-1]
    if lo < 0 or hi >= len(sentences):
        return None
    pos = 0
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    for idx in range(lo, hi + 1):
        sent = sentences[idx]
        j = abstract.find(sent, pos)
        if j == -1:
            j = abstract.find(sent)
            if j == -1:
                return None
        if idx == lo:
            start_char = j
        end_char = j + len(sent)
        pos = end_char
    if start_char is None or end_char is None:
        return None
    return start_char, end_char, abstract[start_char:end_char]


def snippet_from_abstract_window(
    abstract: str,
    sent_ids: List[int],
    allow_fallback: bool,
) -> Optional[Tuple[str, str, str, int, int]]:
    """
    Returns (begin_section, end_section, text, off_begin, off_end) for abstract snippet,
    or None if impossible without fallback (and allow_fallback is False).
    """
    sents = abstract_sentences(abstract)
    if not sents:
        return None
    span = span_for_sentence_range(abstract, sents, sent_ids)
    if span is not None:
        a0, a1, text = span
        return ("abstract", "abstract", text, a0, a1)
    if not allow_fallback:
        return None
    ids = sorted({int(i) for i in sent_ids if i is not None})
    parts = [sents[i] for i in ids if 0 <= i < len(sents)]
    text = ". ".join(parts)
    text = _normalize_unicode_whitespace(text)
    logger.warning(
        "Abstract alignment failed; using joined sentences and fallback offsets (len=%d)",
        len(text),
    )
    return ("abstract", "abstract", text, 0, len(text))


def snippet_from_title(title: str) -> Tuple[str, str, str, int, int]:
    t = title.strip()
    t_norm = _normalize_unicode_whitespace(t)
    return ("title", "title", t_norm, 0, len(t_norm))


def pubmed_document_url(doc_id: str) -> str:
    s = str(doc_id).strip()
    if s.isdigit():
        return f"{PUBMED_DOCUMENT_PREFIX}{s}"
    return s


def contexts_by_doc_keys(contexts: List[dict]) -> Dict[str, dict]:
    """Map doc_id string and/or legacy ``doc`` URL to context row."""
    m: Dict[str, dict] = {}
    for ctx in contexts:
        did = str(ctx.get("doc_id") or "").strip()
        if did:
            m[did] = ctx
        legacy = str(ctx.get("doc") or "").strip()
        if legacy:
            m[legacy] = ctx
            p = pmid_from_url(legacy)
            if p and p not in m:
                m[p] = ctx
    return m


def convert_question(
    q: dict,
    pmid_to_ta: Dict[str, Tuple[str, str]],
    allow_fallback: bool,
) -> dict:
    qid = str(q.get("id", ""))
    doc_ids_in = ranked_doc_ids_from_question(q)[:DOC_CAP]
    docs_out_urls = [pubmed_document_url(d) for d in doc_ids_in]
    ctx_list = q.get("contexts") or []
    by_key = contexts_by_doc_keys(ctx_list if isinstance(ctx_list, list) else [])

    snippets: List[dict] = []
    for doc_id in doc_ids_in:
        doc_s = str(doc_id).strip()
        pmid = doc_s if doc_s.isdigit() else (pmid_from_url(doc_s) or "")
        if not pmid:
            continue
        url_out = pubmed_document_url(doc_s) if doc_s.isdigit() else doc_s
        ta = pmid_to_ta.get(pmid)
        if ta is None:
            logger.warning("Missing corpus row for PMID %s (question %s)", pmid, qid)
            continue
        title, abstract = ta
        ctx = by_key.get(doc_s) or by_key.get(pmid) or by_key.get(url_out)
        if ctx is None:
            continue
        sw = ctx.get("selected_windows") or []
        if not isinstance(sw, list):
            sw = []
        if len(sw) == 0:
            sec_b, sec_e, text, ob, oe = snippet_from_title(title)
            snippets.append(
                {
                    "beginSection": sec_b,
                    "endSection": sec_e,
                    "text": text,
                    "document": url_out,
                    "offsetInBeginSection": ob,
                    "offsetInEndSection": oe,
                }
            )
            continue
        win = pick_best_window(sw)
        if win is None:
            continue
        sids = win.get("sent_ids") or []
        if not isinstance(sids, list) or not sids:
            logger.warning(
                "qid=%s pmid=%s: selected_windows without sent_ids; skip snippet",
                qid,
                pmid,
            )
            continue
        try:
            sids_int = [int(x) for x in sids]
        except (TypeError, ValueError):
            logger.warning("qid=%s pmid=%s: invalid sent_ids; skip snippet", qid, pmid)
            continue
        ab = snippet_from_abstract_window(abstract, sids_int, allow_fallback)
        if ab is None:
            logger.warning(
                "No abstract snippet for qid=%s pmid=%s (alignment failed; use --allow-fallback-offsets or skip)",
                qid,
                pmid,
            )
            continue
        sec_b, sec_e, text, ob, oe = ab
        snippets.append(
            {
                "beginSection": sec_b,
                "endSection": sec_e,
                "text": text,
                "document": url_out,
                "offsetInBeginSection": ob,
                "offsetInEndSection": oe,
            }
        )

    return {
        "id": q.get("id"),
        "type": q.get("type"),
        "body": q.get("body"),
        "documents": docs_out_urls,
        "snippets": snippets,
    }


def load_contexts_questions(path: Path) -> List[dict]:
    """Load question dicts from JSONL (preferred) or legacy JSON ``{\"questions\": [...]}``."""
    if path.suffix.lower() == ".jsonl":
        out: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    out.append(rec)
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions") or []
    if not isinstance(questions, list):
        return []
    return [q for q in questions if isinstance(q, dict)]


def collect_pmids(questions: List[dict]) -> Set[str]:
    """Corpus lookup keys (PubMed PMID strings) from ``doc_ids`` / legacy ``documents``."""
    s: Set[str] = set()
    for q in questions:
        for d in ranked_doc_ids_from_question(q)[:DOC_CAP]:
            ds = str(d).strip()
            if ds.isdigit():
                s.add(ds)
            else:
                p = pmid_from_url(ds)
                if p:
                    s.add(p)
    return s


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        _ensure_nltk_punkt()
        import nltk

        nltk.sent_tokenize("Hello world.")
    except Exception as e:
        logger.error(
            "NLTK is required (same as snippet context build). Install with: pip install nltk. (%s)",
            e,
        )
        return 1
    if not args.contexts_path.is_file():
        logger.error("Contexts file not found: %s", args.contexts_path)
        return 1
    questions = load_contexts_questions(args.contexts_path)
    if not questions:
        logger.error("No question records found in %s", args.contexts_path)
        return 1

    needed = collect_pmids(questions)
    logger.info("Loading corpus for %d PMIDs", len(needed))
    pmid_to_ta = build_pmid_to_title_abstract(args.corpus_path, needed)
    missing = needed - set(pmid_to_ta.keys())
    if missing:
        logger.warning("%d PMIDs missing from corpus (snippets may be fewer)", len(missing))

    out_questions = [
        convert_question(q, pmid_to_ta, args.allow_fallback_offsets)
        for q in questions
        if isinstance(q, dict)
    ]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"questions": out_questions}, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d questions -> %s", len(out_questions), args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
