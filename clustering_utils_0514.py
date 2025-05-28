# clustering_utils.py — Automatic collocate clustering with Longman & vector fallback
# =============================================================================
"""Utility module to cluster collocates and give human‑readable labels.

Highlights
* **Longman Lexicon first**: if a collocate is in `data.longman.lexicon.txt`, we
  group by its Longman category (e.g. "A070 Birds").  Category codes come from
  `data.longman.lexicon.cat.txt`.
* **Vector fallback**: words missing from Longman are clustered with either
  **Affinity Propagation** (default) or **HDBSCAN** — no manual `n_clusters`.
* **`cluster_and_name()`**: convenience wrapper that returns
  `OrderedDict[label] -> [collocates…]` ready for Streamlit.
* **`make_full_graph()`**: builds a PyVis graph for interactive exploration.

The module is import‑safe and also runnable from CLI:

    python clustering_utils.py big accident damage development \
        --cat data.longman.lexicon.cat.txt \
        --lex data.longman.lexicon.txt \
        --algo hdbscan
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import spacy
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network

import re

def readable_label(token: str) -> str:
    code = _norm_code(token)
    m = re.match(r"misc-(?:ap|hdbscan)-(\d+)", token)
    if m:
        return f"Concept {m.group(1)}"
    if code and code in _CODE2LABEL:
        return _CODE2LABEL[code].title()
    if " " in token:
        return token.split(None, 1)[1].title()
    return token.title()

# ---------------------------------------------------------------------------
# 1. spaCy model loader with graceful fallbacks -----------------------------
# ---------------------------------------------------------------------------
from spacy.cli import download as spacy_download

def get_nlp(model: str = "en_core_web_lg"):
    try:
        return spacy.load(model, disable=["tagger", "parser", "ner"])
    except Exception:
        try:
            spacy_download(model)
            return spacy.load(model, disable=["tagger", "parser", "ner"])
        except Exception:
            print("[WARN] Falling back to blank English model – clustering quality may degrade.")
            return spacy.blank("en")

_nlp = get_nlp()
_blank_vec = np.random.RandomState(42).randn(_nlp.vocab.vectors_length or 300)

def vec(token: str) -> np.ndarray:
    lex = _nlp.vocab[token]
    if lex.has_vector:
        return lex.vector
    return _blank_vec

# ---------------------------------------------------------------------------
# 2. Longman Lexicon  +  hierarchy helper
# ---------------------------------------------------------------------------

CODE_RX = re.compile(r"^([A-Z])(\d{1,3})$")   # 1–3 digits

def _norm_code(raw: str) -> str | None:
    m = CODE_RX.match(raw.upper())
    if not m:
        return None
    letter, digits = m.groups()
    return f"{letter}{int(digits):03d}"        # zero-pad to 3 digits

def load_longman(cat_path: str | Path, lex_path: str | Path):
    code2label = {}
    with open(cat_path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if not raw.strip():
                continue
            code_tok, *label_parts = raw.strip().split('\t')
            code = _norm_code(code_tok)
            if not code:
                continue
            code2label[code] = (label_parts[0] if label_parts else "").strip()

    word2codes = defaultdict(list)
    with open(lex_path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if not raw.strip():
                continue
            # Use the full first token as the key (can be a phrase)
            parts = re.split(r"\t", raw.strip())
            # print(parts)
            word = parts[0].lower()
            for tok in parts[1:]:
                code = _norm_code(tok)
                if code and code in code2label:
                    word2codes[word].append(code)
    return code2label, word2codes

_default_cat = Path("data.longman.lexicon.cat.txt")
_default_lex = Path("data.longman.lexicon.txt")
if _default_cat.exists() and _default_lex.exists():
    _CODE2LABEL, _WORD2CODES = load_longman(_default_cat, _default_lex)
else:
    _CODE2LABEL, _WORD2CODES = {}, {}

# ---------------------------------------------------------------------------
# 3. Core clustering helpers -------------------------------------------------
# ---------------------------------------------------------------------------

def _affprop_cluster(vectors: np.ndarray, pref_multiplier: float = -0.5):
    sims = cosine_similarity(vectors)
    pref = np.median(sims) * pref_multiplier
    af = AffinityPropagation(affinity="precomputed", preference=pref, random_state=42)
    labels = af.fit_predict(sims)
    return labels

def _hdbscan_cluster(vectors: np.ndarray):
    try:
        import hdbscan  # type: ignore
    except ImportError:
        raise RuntimeError("hdbscan is not installed – pip install hdbscan")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    return clusterer.fit_predict(vectors)

def _vector_cluster(words: Sequence[str], algorithm: str = "ap"):
    vecs = np.stack([vec(w) for w in words])
    if algorithm == "hdbscan":
        labels = _hdbscan_cluster(vecs)
    else:
        labels = _affprop_cluster(vecs)
    out: Dict[int, List[str]] = defaultdict(list)
    for w, lab in zip(words, labels):
        out[int(lab)].append(w)
    return list(out.values())

# ---------------------------------------------------------------------------
# 4. Public API -------------------------------------------------------------
# ---------------------------------------------------------------------------

def cluster_collocates(collocates: Sequence[str], *,
                       prefer_longman: bool = True,
                       algorithm: str = "ap",
                       cat_file: str | Path | None = None,
                       lex_file: str | Path | None = None) -> "OrderedDict[str, List[str]]":
    """Return OrderedDict[label] = [words…].

    * Longman categories first (if available & prefer_longman=True).
    * Remaining words clustered by vectors (AP or HDBSCAN).
    """
    # 1️⃣  Longman pass
    if prefer_longman:
        # print("here")
        code2label, word2codes = (_CODE2LABEL, _WORD2CODES)
        # if cat_file and lex_file:
        code2label, word2codes = load_longman(cat_file, lex_file)
        # print(word2codes['big'])
    else:
        code2label, word2codes = {}, {}

    used = set()
    groups: OrderedDict[str, List[str]] = OrderedDict()

    # Only exact match for single words or phrases
    for w in collocates:
        key = w.lower()
        # print(f"Trying to match: '{key}'")
        # if key in word2codes:
        #     print(f"Matched in lexicon: {key} -> {word2codes[key]}")
        # else:
        #     print(f"No match for: {key}")
        if key in word2codes and word2codes[key]:
            for code in word2codes[key]:
                label = code2label.get(code, code)
                groups.setdefault(label, []).append(w)
            used.add(w)

    # 2️⃣  Vector clustering for leftovers
    leftovers = [w for w in collocates if w not in used]
    if leftovers:
        clusters = _vector_cluster(leftovers, algorithm=algorithm)
        for idx, words in enumerate(clusters):
            if words:
                groups[f"misc-{algorithm}-{idx}"] = words

    return groups

def cluster_and_name(collocates: Sequence[str], **kwargs):
    """Alias keeping backwards compatibility."""
    return cluster_collocates(collocates, **kwargs)

# ---------------------------------------------------------------------------
# 5. PyVis full graph -------------------------------------------------------
# ---------------------------------------------------------------------------

def make_full_graph(error_data: Dict[str, Dict], *, height="700px", width="100%"):
    G = nx.Graph()
    palette: Dict[str, str] = {}
    colour_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                    "#bcbd22", "#17becf"]

    for err, info in error_data.items():
        # Replacement nodes
        for rep, r_info in info["replacements"].items():
            rep_node = f"{err} → {rep}"
            G.add_node(rep_node, color="#812503", shape="box",
                       title=f"{r_info['count']} corrections")

        # Collocate nodes
        for label, collocs in info["clusters"].items():
            nice = readable_label(label)
            colour = palette.setdefault(nice, colour_cycle[len(palette) % len(colour_cycle)])
            for c in collocs:
                G.add_node(c, color=colour, title=nice, group=nice)

        # Edges
        for rep, r_info in info["replacements"].items():
            rep_node = f"{err} → {rep}"
            for c in list(r_info["collocations"].keys()):
                G.add_edge(rep_node, c, title=f"{rep} ↔ {c}")

    net = Network(height=height, width=width, bgcolor="#222", font_color="white")
    net.from_nx(G)
    return net, palette

# ---------------------------------------------------------------------------
# 6. CLI entry --------------------------------------------------------------
# ---------------------------------------------------------------------------
def _cli():
    import argparse, json, sys

    ap = argparse.ArgumentParser(
        description="Cluster the provided collocates and print JSON.")
    ap.add_argument("words", nargs="+", help="Collocate strings to cluster")
    ap.add_argument("--algo", choices=["ap", "hdbscan"], default="ap",
                    help="Vector clustering algorithm (default: ap)")
    ap.add_argument("--cat", dest="cat_file", default="data.longman.lexicon.cat.txt")
    ap.add_argument("--lex", dest="lex_file", default="data.longman.lexicon.txt")
    ap.add_argument("--no-longman", dest="use_longman", action="store_false",
                    help="Skip Longman mapping, use vectors only")

    args = ap.parse_args()
    clusters = cluster_collocates(
        args.words,
        prefer_longman=args.use_longman,
        algorithm=args.algo,
        cat_file=args.cat_file,
        lex_file=args.lex_file,
    )
    json.dump(clusters, sys.stdout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    _cli()