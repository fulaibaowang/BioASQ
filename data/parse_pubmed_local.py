#!/usr/bin/env python3
"""
Parse local PubMed *.xml.gz files into JSONL shards (1 xml.gz -> 1 jsonl).

Input: a directory containing .xml.gz files (e.g. baseline/ or updatefiles/)
Output: a directory for .jsonl files, same filenames except .jsonl

Extracted fields per line:
- pmid (str)
- docno (str)  # same as pmid, PyTerrier-friendly
- title (str)
- abstract (str)
- mesh_terms (str)    # "Dxxxx:Term; Dyyyy:Term" (descriptor-only)
- keywords (list[str])
- is_deleted (bool)   # True for DeleteCitation tombstones

Requirements:
    pip install lxml pubmed-parser
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def parse_pmid(medline: etree._Element) -> str:
    pmid = medline.find("PMID")
    if pmid is not None and pmid.text:
        return pmid.text.strip()
    # fallback
    article_ids = medline.find("PubmedData/ArticleIdList")
    if article_ids is not None:
        x = article_ids.find('ArticleId[@IdType="pubmed"]')
        if x is not None and x.text:
            return x.text.strip()
    return ""


def parse_keywords(medline: etree._Element) -> List[str]:
    if pp is not None:
        try:
            kws = pp.medline_parser.parse_keywords(medline)
            # pubmed_parser sometimes returns None
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
    pmid = parse_pmid(medline)
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


def iter_records_from_xml_gz(gz_path: Path) -> Iterable[Dict]:
    """
    Yields MedlineCitation records and DeleteCitation tombstones.
    DeleteCitation mostly appears in updatefiles.
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    with gzip.open(gz_path, "rb") as fh:
        for _, elem in etree.iterparse(fh, events=("end",)):
            if elem.tag == "MedlineCitation":
                rec = parse_article_record(elem)
                if rec is not None:
                    yield rec
                elem.clear()

            elif elem.tag == "DeleteCitation":
                for pmid_node in elem.findall("PMID"):
                    if pmid_node.text and pmid_node.text.strip():
                        pmid = pmid_node.text.strip()
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


def xml_gz_to_jsonl(gz_path: Path, jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = jsonl_path.with_suffix(".jsonl.partial")
    with open(tmp, "w", encoding="utf-8") as out:
        for rec in iter_records_from_xml_gz(gz_path):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing *.xml.gz files (baseline or updatefiles).")
    ap.add_argument("--output_dir", required=True, help="Folder to write *.jsonl shards.")
    ap.add_argument("--glob", default="*.xml.gz", help="Glob pattern. Default: *.xml.gz")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if corresponding JSONL exists.")
    ap.add_argument("--max_files", type=int, default=None, help="Parse only first N files (debug).")
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
        xml_gz_to_jsonl(gz_path, jsonl_path)

    print("[DONE]")


if __name__ == "__main__":
    main()
