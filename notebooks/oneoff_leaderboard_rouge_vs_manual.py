# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Does ROUGE track the manual score on the BioASQ leaderboard?
#
# One-off analysis of the **public** BioASQ results pages (no pipeline output involved). For each
# test batch, every system gets one point: its ROUGE score against its mean manual score (the mean
# of readability, recall, precision and repetition, each 1–5). We ask whether the points fall on a
# line or in a cloud, and report Spearman and Pearson correlations.
#
# Two filters matter, because without them the correlation is easy to overstate:
#
# - **manual ≥ 3** — each batch has a few systems near manual 1 / ROUGE 0. They separate from the
#   main pack on both axes and would carry the correlation on their own.
# - **one system per team** — a team's submissions score alike, so a team with five runs would
#   count five times. We keep each team's system with the highest mean manual score.
#
# The results pages list no team, so **teams are inferred from system names** (`TEAM_RULES` below).
# Where a grouping was uncertain we merged rather than split.
#
# Tables are cached as CSV under `leaderboard_rouge_vs_manual/tables/`; delete a file to re-fetch it.
# Scores on the pages can still change after a challenge ends, so the CSVs are a snapshot.

# %%
from __future__ import annotations

import csv
import html
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# %% [markdown]
# ## 1) Paths and config

# %%
try:
    _NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:
    _NOTEBOOK_DIR = Path.cwd()
OUT_DIR = _NOTEBOOK_DIR / "leaderboard_rouge_vs_manual"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"

RESULTS_URL = "https://participants-area.bioasq.org/results/{task}/{phase}/"
# name -> (task, phase) as they appear in the results URL
DATASETS: Dict[str, Tuple[str, str]] = {
    "14b_phaseB": ("14b", "phaseB"),
    "13b_phaseB": ("13b", "phaseB"),
    "12b_phaseB": ("12b", "phaseB"),
    "11b_phaseB": ("11b", "phaseB"),
}
DATASET_TITLES = {
    "14b_phaseB": "BioASQ 14b Phase B",
    "13b_phaseB": "BioASQ 13b Phase B",
    "12b_phaseB": "BioASQ 12b Phase B",
    "11b_phaseB": "BioASQ 11b Phase B",
}
BATCHES = [1, 2, 3, 4]

MANUAL_COLS = ["Readability", "Recall", "Precision", "Repetition"]
ROUGE_COLS = ["R-2 (F1)", "R-2 (Rec)", "R-SU4 (F1)", "R-SU4 (Rec)"]
ROUGE_LABELS = {
    "R-2 (F1)": "ROUGE-2 F1",
    "R-2 (Rec)": "ROUGE-2 recall",
    "R-SU4 (F1)": "ROUGE-SU4 F1",
    "R-SU4 (Rec)": "ROUGE-SU4 recall",
}
MANUAL_MIN = 3.0

# Team is inferred from the system name. First matching pattern wins; a name that matches nothing
# is its own team. Uncertain groupings are merged (IR_J with IR_Y, the Final*/qwen runs,
# h-nlp-* with health-nlp, the deepseek and GPT runs of 13b, the Gatech runs of 12b).
TEAM_RULES: List[Tuple[str, str]] = [
    (r'^"?RMC_', "RMC"),
    (r"^\d system$", "N system"),
    (r"^13b[-_]", "13b"),
    (r"DMIS-KU", "DMIS-KU"),
    (r"^AsqAway_", "AsqAway"),
    (r"^AUEB-System", "AUEB"),
    (r"^BioASQ[ _]Baseline", "BioASQ Baseline"),
    (r"^bioinfo-", "bioinfo"),
    (r"^Biomedical QA", "Biomedical QA"),
    (r"^bious\d", "bious"),
    (r"^capstone-", "capstone"),
    (r"^config-", "config"),
    (r"^CPS\d?$", "CPS"),
    (r"^CSA-IISR", "CSA-IISR"),
    (r"^IISR[ -]", "IISR"),
    (r"^deepseek", "deepseek"),
    (r"^dictycite", "dictycite"),
    (r"^DICE", "DICE"),
    (r"^dmiip", "dmiip"),
    (r"^DMIS_MES_TEST", "DMIS_MES_TEST"),
    (r"^ku_dmis", "ku_dmis"),
    (r"^(DS@GT|DSGT)", "DSGT"),
    (r"^ELErank", "ELErank"),
    (r"^EP-", "EP"),
    (r"^Fleming-", "Fleming"),
    (r"^(Finalcorrected|FinalQwen|qwen)$", "Final/qwen"),
    (r"^(Gatech|GTBioASQ)", "Gatech"),
    (r"^(?i:gpt)", "GPT"),
    (r"^(health-nlp|h-nlp)", "health-nlp"),
    (r"^IBE-LM", "IBE-LM"),
    (r"^IR_[JY]-", "IR_J/IR_Y"),
    (r"^(sp_)?(?i:lasige)", "lasige"),
    (r"^lean_rag", "lean_rag"),
    (r"^mckpt", "mckpt"),
    (r"^MedQA-", "MedQA"),
    (r"^mibi_rag", "mibi_rag"),
    (r"^MindLab", "MindLab"),
    (r"^MQ-", "MQ"),
    (r"^(?i:mistral)", "mistral"),
    (r"^multi-stage rank", "multi-stage rank&llm"),
    (r"^pancras_", "pancras"),
    (r"^phaseB-\d", "phaseB"),
    (r"^UniTor_", "UniTor"),
    (r"^UR-IW-", "UR-IW"),
    (r"_&_LLM$", "*_&_LLM"),
]

POINT_COLOR = "#444444"  # monochrome: one series, nothing to tell apart by hue


def team_of(system: str) -> str:
    for pattern, team in TEAM_RULES:
        if re.search(pattern, system):
            return team
    return system


# %% [markdown]
# ## 2) Fetch the results pages into CSV
#
# Each page holds two tables per batch, in order: exact answers, then ideal answers. The first
# header row only groups columns, so the second one is the real header.
#
# The 11b page lists a fifth "batch" that is a copy of 12b batch 1, so only the first four batches
# of a page are kept.

# %%
def _cell_text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def fetch_tables(task: str, phase: str) -> List[List[List[str]]]:
    req = urllib.request.Request(RESULTS_URL.format(task=task, phase=phase), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    tables = []
    for m in re.finditer(r"<table.*?>(.*?)</table>", page, re.S):
        rows = [
            [_cell_text(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
            for r in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S)
        ]
        tables.append(rows[1:])
    return tables


def table_path(dataset: str, batch: int, kind: str) -> Path:
    return TABLES_DIR / f"{dataset}_batch{batch}_{kind}.csv"


def ensure_tables(dataset: str) -> None:
    wanted = [table_path(dataset, b, k) for b in BATCHES for k in ("exact", "ideal")]
    if all(p.exists() for p in wanted):
        return
    tables = fetch_tables(*DATASETS[dataset])
    if len(tables) < len(wanted):
        raise RuntimeError(f"{dataset}: expected {len(wanted)} tables on the page, found {len(tables)}")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(wanted, tables):
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
    print(f"{dataset}: fetched {len(wanted)} tables")


for _dataset in DATASETS:
    ensure_tables(_dataset)

# %% [markdown]
# ## 3) Load, filter, correlate
#
# A system with `-` in the ROUGE column or in any manual column is dropped for that batch.

# %%
Point = Tuple[str, float, float]  # system, rouge, mean manual


def _num(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def load_points(dataset: str, batch: int, rouge_col: str) -> List[Point]:
    points = []
    with table_path(dataset, batch, "ideal").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals = [_num(row[c]) for c in [rouge_col] + MANUAL_COLS]
            if None in vals:
                continue
            points.append((row["System"], vals[0], sum(vals[1:]) / len(MANUAL_COLS)))
    return points


def best_per_team(points: List[Point]) -> Dict[str, Point]:
    """Each team's system with the highest mean manual score."""
    by_team: Dict[str, List[Point]] = defaultdict(list)
    for p in points:
        by_team[team_of(p[0])].append(p)
    return {team: max(ps, key=lambda p: p[2]) for team, ps in by_team.items()}


def all_systems(points: List[Point]) -> List[Point]:
    return points


def systems_manual_min(points: List[Point]) -> List[Point]:
    return [p for p in points if p[2] >= MANUAL_MIN]


def one_per_team_manual_min(points: List[Point]) -> List[Point]:
    return [p for p in best_per_team(points).values() if p[2] >= MANUAL_MIN]


FILTERS: Dict[str, Callable[[List[Point]], List[Point]]] = {
    "all_systems": all_systems,
    "manual_ge_3": systems_manual_min,
    "one_per_team_manual_ge_3": one_per_team_manual_min,
}
FILTER_NOTES = {
    "all_systems": ("one dot per system", "Manual score: mean of readability, recall, precision and repetition."),
    "manual_ge_3": ("one dot per system", "Systems with a mean manual score below 3 are left out."),
    "one_per_team_manual_ge_3": (
        "one dot per team",
        "Each team's best system (mean manual score ≥ 3); teams inferred from system names.",
    ),
}


def correlate(points: List[Point]) -> Dict[str, float]:
    x = [p[1] for p in points]
    y = [p[2] for p in points]
    rho, rho_p = spearmanr(x, y)
    r, r_p = pearsonr(x, y)
    return {"n": len(points), "spearman_rho": rho, "spearman_p": rho_p, "pearson_r": r, "pearson_p": r_p}


# %% [markdown]
# ## 4) Correlation table
#
# Every dataset × batch × ROUGE variant × filter. Written to `correlations.csv`; `teams_picked.csv`
# records which system stands in for each team, so the team inference can be audited.

# %%
OUT_DIR.mkdir(parents=True, exist_ok=True)

correlation_rows = []
for dataset in DATASETS:
    for batch in BATCHES:
        for rouge_col in ROUGE_COLS:
            points = load_points(dataset, batch, rouge_col)
            for filter_name, keep in FILTERS.items():
                stats = correlate(keep(points))
                correlation_rows.append({
                    "dataset": dataset,
                    "batch": batch,
                    "rouge": rouge_col,
                    "filter": filter_name,
                    "n": stats["n"],
                    "spearman_rho": f"{stats['spearman_rho']:.3f}",
                    "spearman_p": f"{stats['spearman_p']:.2g}",
                    "pearson_r": f"{stats['pearson_r']:.3f}",
                    "pearson_p": f"{stats['pearson_p']:.2g}",
                })

with (OUT_DIR / "correlations.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(correlation_rows[0]))
    writer.writeheader()
    writer.writerows(correlation_rows)

with (OUT_DIR / "teams_picked.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "batch", "team", "n_systems", "picked_system", "R-2 (F1)", "manual_mean"])
    for dataset in DATASETS:
        for batch in BATCHES:
            points = load_points(dataset, batch, "R-2 (F1)")
            sizes: Dict[str, int] = defaultdict(int)
            for p in points:
                sizes[team_of(p[0])] += 1
            for team, p in sorted(best_per_team(points).items(), key=lambda kv: -kv[1][2]):
                writer.writerow([dataset, batch, team, sizes[team], p[0], f"{p[1]:.4f}", f"{p[2]:.4f}"])

print(f"{'dataset':16s} {'batch':>5s}  " + "  ".join(f"{name:>26s}" for name in FILTERS))
for dataset in DATASETS:
    for batch in BATCHES:
        cells = []
        for filter_name in FILTERS:
            row = next(
                r for r in correlation_rows
                if (r["dataset"], r["batch"], r["rouge"], r["filter"]) == (dataset, batch, "R-2 (F1)", filter_name)
            )
            cells.append(f"rho={row['spearman_rho']} (n={row['n']})".rjust(26))
        print(f"{dataset:16s} {batch:>5d}  " + "  ".join(cells))

# %% [markdown]
# ## 5) Plots
#
# One 2×2 figure per dataset and filter, one panel per batch. The filtered figures start the
# manual axis at 3, so their spread looks larger than in the all-systems figure.

# %%
def plot_dataset(dataset: str, rouge_col: str, filter_name: str) -> Path:
    panels = [FILTERS[filter_name](load_points(dataset, b, rouge_col)) for b in BATCHES]
    unit, note = FILTER_NOTES[filter_name]
    y_min = 0.0 if filter_name == "all_systems" else MANUAL_MIN
    x_max = max(p[1] for pts in panels for p in pts) * 1.08

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6), sharex=True, sharey=True)
    for batch, points, ax in zip(BATCHES, panels, axes.flat):
        stats = correlate(points)
        ax.scatter([p[1] for p in points], [p[2] for p in points],
                   s=40, color=POINT_COLOR, alpha=0.7, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(f"Batch {batch}", loc="left", fontsize=12, fontweight="bold")
        p_txt = "p < 0.001" if stats["spearman_p"] < 0.001 else f"p = {stats['spearman_p']:.3f}"
        ax.text(0.97, 0.05,
                f"Spearman ρ = {stats['spearman_rho']:.2f} ({p_txt})\n"
                f"Pearson r = {stats['pearson_r']:.2f}\n"
                f"n = {stats['n']} {'teams' if unit.endswith('team') else 'systems'}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, linespacing=1.5)
        ax.grid(color="#e6e6e6", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].set_xlim(0, x_max)
    axes.flat[0].set_ylim(y_min, 5.05 if y_min else 5.2)
    y_label = "Mean manual score (1–5)" if not y_min else f"Mean manual score (axis starts at {y_min:g})"
    for ax in axes[1]:
        ax.set_xlabel(ROUGE_LABELS[rouge_col])
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)
    fig.suptitle(f"{DATASET_TITLES[dataset]} ideal answers: {ROUGE_LABELS[rouge_col]} vs manual score, {unit}",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.95, note, fontsize=9.5, va="top")
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.92))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    slug = rouge_col.replace("R-", "rouge").replace(" (", "_").replace(")", "").lower()
    out = FIGURES_DIR / f"{dataset}_{slug}_{filter_name}.png"
    fig.savefig(out, dpi=170)
    return out


for dataset in DATASETS:
    for rouge_col in ("R-2 (F1)", "R-2 (Rec)"):
        for filter_name in ("all_systems", "one_per_team_manual_ge_3"):
            print(plot_dataset(dataset, rouge_col, filter_name).relative_to(_NOTEBOOK_DIR))
            plt.show()
            plt.close("all")
