"""Utility to download and prepare the Cora dataset vocabulary.

The Cora vocabulary is the list of 1433 unique words used to construct
the bag-of-words features. This script downloads it from known public
sources so that paper text can be reconstructed for semantic embedding.
"""

from __future__ import annotations

import logging
import os
import urllib.request

LOGGER = logging.getLogger(__name__)

# Known mirrors for the Cora vocabulary (1433 words, one per line).
VOCAB_URLS = [
    "https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/cora_vocab.txt",
    "https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/cora_vocab.txt",
    "https://raw.githubusercontent.com/Teichlab/sciboro/main/data/cora_vocab.txt",
]

LOCAL_PATH = "./data/cora_vocab.txt"


def download_vocabulary(output_path: str = LOCAL_PATH) -> bool:
    """Try to download the Cora vocabulary from known mirrors.

    Returns True if a valid 1433-word vocabulary was saved.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for url in VOCAB_URLS:
        try:
            LOGGER.info("Trying: %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if 1400 <= len(lines) <= 1450:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                LOGGER.info(
                    "Downloaded %d words to %s from %s",
                    len(lines),
                    output_path,
                    url,
                )
                return True
            LOGGER.warning(
                "Got %d words from %s, expected ~1433. Trying next mirror.",
                len(lines),
                url,
            )
        except Exception as exc:
            LOGGER.warning("Failed to download from %s: %s", url, exc)

    LOGGER.error(
        "Could not download Cora vocabulary from any known mirror. "
        "Please manually place a 1433-word vocabulary file at %s",
        output_path,
    )
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    download_vocabulary()
