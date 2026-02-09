#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from lxml import etree

try:
    import pubmed_parser as pp
except Exception:
    pp = None


def _stringify(node: Optional[etree._Element]) -> str:
    if node is None:
        return ""
    if pp is not None:
        try:
            return (pp.utils.stringify_children(node) or "").strip()
        except Exception:
            pass
    return " ".join(t.strip() for t in node.itertext() if t and t.strip()).strip()


def parse_mesh_terms(medline: etree._Element) -> str:
    """Descriptor-only MeSH terms: 'D006973:Hypertension; D011247:Pregnancy'"""
    mesh_list = medline.find("MeshHeadingList")
    if mesh_list is None:
        return ""
    out: List[str] = []
    for mh in mesh_list.findall("MeshHeading"):
        d = mh.find("DescriptorName")
        if d is None:
            continue
        ui = d.attrib.get("UI", "") or ""
        txt = (d.text or "").strip()
        if ui and txt:
            out.append(f"{ui}:{txt}")
        elif txt:
            out.append(txt)
    return "; ".join(out)


def parse_pmid_all(medline: etree._Element) -> str:
    """
    Extract PMID from any version (Version="1" or others).
    Prefers Version="1" if available, falls back to first PMID found.
    """
    # Prefer the Version="1" PMID
    pmid_v1 = medline.find('PMID[@Version="1"]')
    if pmid_v1 is not None and pmid_v1.text:
        return pmid_v1.text.strip()
    
    # Fall back to any PMID element
    pmid_any = medline.find('PMID')
    if pmid_any is not None and pmid_any.text:
        return pmid_any.text.strip()
    
    return ""


def parse_keywords(medline: etree._Element) -> List[str]:
    if pp is not None:
        try:
            kws = pp.medline_parser.parse_keywords(medline)
            return kws if kws is not None else []
        except Exception:
            pass

    kws: List[str] = []
    for kw in medline.findall(".//KeywordList/Keyword"):
        t = (kw.text or "").strip()
        if t:
            kws.append(t)
    return kws


def parse_title_abstract(medline: etree._Element) -> Dict[str, str]:
    article = medline.find("Article")
    if article is None:
        return {"title": "", "abstract": ""}

    title = _stringify(article.find("ArticleTitle"))

    abs_texts = article.findall("Abstract/AbstractText")
    if abs_texts:
        if len(abs_texts) > 1:
            parts: List[str] = []
            for a in abs_texts:
                label = a.attrib.get("Label", "") or a.attrib.get("NlmCategory", "")
                if label and label != "UNASSIGNED":
                    parts.append(label)
                parts.append(_stringify(a))
            abstract = "\n".join([p for p in parts if p]).strip()
        else:
            abstract = _stringify(abs_texts[0])
    else:
        abstract = _stringify(article.find("Abstract"))

    return {"title": title, "abstract": abstract}


def parse_article_record(medline: etree._Element) -> Optional[Dict]:
    pmid = parse_pmid_all(medline)
    if not pmid:
        return None

    ta = parse_title_abstract(medline)
    return {
        "pmid": pmid,
        "docno": pmid,  # PyTerrier-friendly
        "title": ta["title"],
        "abstract": ta["abstract"],
        "mesh_terms": parse_mesh_terms(medline),
        "keywords": parse_keywords(medline),
        "is_deleted": False,
    }


def iter_records_from_xml_gz(gz_path: Path, dedup: bool = True) -> Iterable[Dict]:
    """
    Yields MedlineCitation records (all PMID versions), plus DeleteCitation tombstones.
    Dedup: ensures each PMID appears at most once in this xml.gz output.
    """
    seen: Set[str] = set()

    with gzip.open(gz_path, "rb") as fh:
        for _, elem in etree.iterparse(fh, events=("end",)):
            if elem.tag == "MedlineCitation":
                rec = parse_article_record(elem)
                elem.clear()
                if rec is None:
                    continue
                pmid = rec["pmid"]
                if dedup:
                    if pmid in seen:
                        continue
                    seen.add(pmid)
                yield rec

            elif elem.tag == "DeleteCitation":
                # Updatefiles only. Keep as tombstone.
                for pmid_node in elem.findall("PMID"):
                    if pmid_node.text and pmid_node.text.strip():
                        pmid = pmid_node.text.strip()
                        if dedup:
                            if pmid in seen:
                                continue
                            seen.add(pmid)
                        yield {
                            "pmid": pmid,
                            "docno": pmid,
                            "title": "",
                            "abstract": "",
                            "mesh_terms": "",
                            "keywords": [],
                            "is_deleted": True,
                        }
                elem.clear()

            else:
                # memory hygiene
                if len(elem) > 1000:
                    elem.clear()


def xml_gz_to_jsonl(gz_path: Path, jsonl_path: Path, dedup: bool = True) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = jsonl_path.with_suffix(".jsonl.partial")
    with open(tmp, "w", encoding="utf-8") as out:
        for rec in iter_records_from_xml_gz(gz_path, dedup=dedup):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing *.xml.gz files (baseline or updatefiles).")
    ap.add_argument("--output_dir", required=True, help="Folder to write *.jsonl shards.")
    ap.add_argument("--glob", default="*.xml.gz")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--no_dedup", action="store_true", help="Disable per-file PMID dedup (not recommended).")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"[INFO] Found {len(files)} files in {in_dir}")

    for gz_path in files:
        jsonl_path = out_dir / gz_path.name.replace(".xml.gz", ".jsonl")
        if args.skip_existing and jsonl_path.exists() and jsonl_path.stat().st_size > 0:
            print(f"[SKIP] {gz_path.name}")
            continue
        print(f"[PARSE] {gz_path.name} -> {jsonl_path.name}")
        xml_gz_to_jsonl(gz_path, jsonl_path, dedup=(not args.no_dedup))

    print("[DONE]")


if __name__ == "__main__":
    main()
