"""Legacy entry point for semantic feature extraction."""

from __future__ import annotations

import argparse
import logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract Text Attributes using LLM")
    parser.add_argument("--dataset", type=str, default="Cora")
    return parser.parse_args()


def main() -> None:
    """Report the legacy extraction entry point."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    LOGGER.info("dataset=%s status=legacy_entry_point", args.dataset)
    LOGGER.info("Use 01_extract_llm.py for the reproducible Cora anchor pipeline.")


if __name__ == "__main__":
    main()