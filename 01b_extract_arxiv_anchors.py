"""Generate real semantic anchors for OGB-ArXiv using sentence-transformers.

Uses the downloaded title+abstract text and maps them to node indices
via the OGB nodeidx2paperid mapping. The resulting anchors capture
genuine paper content semantics, independent of the citation graph.

Expected runtime: ~5 min on GPU, ~30 min on CPU for 169K papers.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import random

import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_title_abstract(tsv_path: str) -> dict[str, str]:
    """Load paper_id -> title+abstract mapping from gzipped TSV.

    The TSV format is: paper_id \\t title \\t abstract
    First 3 lines are tar header junk, skipped automatically.
    """
    LOGGER.info("Loading title+abstract from %s ...", tsv_path)
    id2text: dict[str, str] = {}
    with gzip.open(tsv_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < 3:
                continue  # skip header lines
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            paper_id, title, abstract = parts[0], parts[1], parts[2]
            # Truncate very long abstracts to avoid OOM during encoding
            text = f"{title}. {abstract}"[:1024]
            id2text[paper_id] = text
    LOGGER.info("Loaded %d paper texts", len(id2text))
    return id2text


def load_paperid_to_nodeidx(mapping_dir: str) -> dict[str, int]:
    """Load mapping from paper MAG ID to node index."""
    path = os.path.join(mapping_dir, "nodeidx2paperid.csv.gz")
    LOGGER.info("Loading nodeidx2paperid from %s ...", path)
    df = pd.read_csv(path, compression="gzip", header=None, names=["node_idx", "paper_id"])
    # Skip header row that got included (file has no proper header)
    df = df[df["paper_id"] != "paper id"]
    df["paper_id"] = df["paper_id"].astype(str)
    df["node_idx"] = df["node_idx"].astype(int)
    mapping = dict(zip(df["paper_id"], df["node_idx"]))
    LOGGER.info("Mapping: %d paper IDs -> node indices", len(mapping))
    return mapping


def generate_anchors(
    id2text: dict[str, str],
    paperid2node: dict[str, str],
    n_nodes: int,
    model_name: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Generate sentence-transformer embeddings for all nodes.

    Nodes without matching text get the mean embedding (rare).
    """
    from sentence_transformers import SentenceTransformer

    LOGGER.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name, device=device)

    # Build ordered text list aligned with node indices
    texts: list[str] = [""] * n_nodes
    matched = 0
    for paper_id, text in id2text.items():
        node_idx = paperid2node.get(paper_id)
        if node_idx is not None and node_idx < n_nodes:
            texts[node_idx] = text
            matched += 1

    LOGGER.info("Matched %d/%d nodes (%.1f%%)", matched, n_nodes, 100 * matched / n_nodes)

    # Fill missing texts with placeholder
    empty_indices = [i for i, t in enumerate(texts) if not t]
    if empty_indices:
        LOGGER.warning("%d nodes have no text, using placeholder", len(empty_indices))
        for i in empty_indices:
            texts[i] = "arxiv paper"

    LOGGER.info("Encoding %d texts (batch_size=%d) ...", n_nodes, batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    LOGGER.info("Embedding shape: %s", tuple(embeddings.shape))
    return embeddings.cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="ArXiv semantic anchor generation")
    parser.add_argument("--tsv_path", type=str,
                        default="./data/arxiv/titleabs.tsv.gz")
    parser.add_argument("--ogb_root", type=str,
                        default="./data/ogb")
    parser.add_argument("--output_path", type=str,
                        default="./embeddings/arxiv_semantic_anchor.pt")
    parser.add_argument("--model_name", type=str,
                        default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load texts
    id2text = load_title_abstract(args.tsv_path)

    # Load paper ID -> node index mapping
    mapping_dir = os.path.join(args.ogb_root, "ogbn_arxiv", "mapping")
    if not os.path.exists(mapping_dir):
        raise FileNotFoundError(
            f"Mapping directory not found: {mapping_dir}. "
            "Run OGB ArXiv dataset download first."
        )
    paperid2node = load_paperid_to_nodeidx(mapping_dir)

    # Determine number of nodes from OGB metadata
    n_nodes = max(paperid2node.values()) + 1
    LOGGER.info("Total nodes: %d", n_nodes)

    # Generate embeddings
    anchors = generate_anchors(id2text, paperid2node, n_nodes,
                               args.model_name, args.batch_size, device)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(anchors, args.output_path)
    LOGGER.info("Saved anchors to %s", args.output_path)


if __name__ == "__main__":
    main()
