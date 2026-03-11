#!/usr/bin/env python3
"""Build pre-computed card text embeddings from cards.cdb.

Reads card effect text from the SQLite database, encodes it using a
sentence-transformer model, and saves the embeddings as a .pt file
for use during training.

Usage:
    scripts/build_card_embeddings.sh
    scripts/build_card_embeddings.sh --db path/to/cards.cdb --output path/to/embeddings.pt
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build card text embeddings from cards.cdb")
    parser.add_argument("--db", type=str, default="assets/cards.cdb",
                        help="Path to cards.cdb SQLite database (default: assets/cards.cdb)")
    parser.add_argument("--output", type=str, default="assets/card_text_embeddings.pt",
                        help="Output path for embeddings .pt file (default: assets/card_text_embeddings.pt)")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"Sentence-transformer model name (default: {MODEL_NAME})")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size (default: 256)")
    return parser.parse_args()


def load_card_texts(db_path: str) -> tuple[list[int], list[str]]:
    """Read (id, desc) pairs from cards.cdb texts table."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT id, desc FROM texts ORDER BY id")
        rows = cursor.fetchall()
    finally:
        conn.close()

    codes = []
    descriptions = []
    for card_id, desc in rows:
        codes.append(card_id)
        descriptions.append(desc if desc else "")

    return codes, descriptions


def main() -> None:
    args = parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "Required packages not installed. Run: pip install -e '.[embed,train]'"
        )
        sys.exit(1)

    logger.info("Loading card texts from %s", db_path)
    codes, descriptions = load_card_texts(str(db_path))
    logger.info("Found %d cards", len(codes))

    logger.info("Loading sentence-transformer model: %s", args.model)
    model = SentenceTransformer(args.model)

    logger.info("Encoding card descriptions (batch_size=%d)...", args.batch_size)
    embeddings = model.encode(
        descriptions,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings_tensor = torch.from_numpy(embeddings).float()
    codes_tensor = torch.tensor(codes, dtype=torch.int64)

    # Zero out embeddings for cards with empty descriptions
    num_empty = 0
    for i, desc in enumerate(descriptions):
        if not desc.strip():
            embeddings_tensor[i] = 0.0
            num_empty += 1
    if num_empty:
        logger.info("Zeroed %d cards with empty descriptions", num_empty)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings_tensor,
        "codes": codes_tensor,
        "model_name": args.model,
    }, str(output_path))

    logger.info(
        "Saved %d embeddings (dim=%d) to %s",
        len(codes), embeddings_tensor.shape[1], output_path,
    )


if __name__ == "__main__":
    main()
