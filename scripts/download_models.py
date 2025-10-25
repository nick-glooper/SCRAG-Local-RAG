#!/usr/bin/env python3
"""
Model downloader for SecureRAG.
Downloads embedding and reranking models to local cache.
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_models(models_dir: Path = None):
    """
    Download required models to local cache.

    Args:
        models_dir: Optional directory to cache models
    """
    if models_dir:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        cache_folder = str(models_dir)
    else:
        cache_folder = None

    models = [
        {
            "name": "Embedding Model",
            "model_id": "BAAI/bge-large-en-v1.5",
            "type": "embedding",
            "size_gb": 1.34
        },
        {
            "name": "Reranking Model",
            "model_id": "BAAI/bge-reranker-large",
            "type": "reranker",
            "size_gb": 1.11
        }
    ]

    print("=" * 60)
    print("SecureRAG Model Downloader")
    print("=" * 60)
    print("\nThis will download the following models:")
    print()

    for model in models:
        print(f"  • {model['name']}")
        print(f"    ID: {model['model_id']}")
        print(f"    Size: ~{model['size_gb']} GB")
        print()

    total_size = sum(m['size_gb'] for m in models)
    print(f"Total download size: ~{total_size:.2f} GB")
    print()

    response = input("Continue with download? [y/N]: ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    print("\nDownloading models...")
    print("=" * 60)

    for model in models:
        print(f"\n📥 Downloading {model['name']}...")
        print(f"   Model: {model['model_id']}")

        try:
            if model['type'] == 'embedding':
                logger.info(f"Loading embedding model: {model['model_id']}")
                SentenceTransformer(
                    model['model_id'],
                    cache_folder=cache_folder
                )
            elif model['type'] == 'reranker':
                logger.info(f"Loading reranker model: {model['model_id']}")
                CrossEncoder(
                    model['model_id'],
                    cache_folder=cache_folder
                )

            print(f"   ✓ {model['name']} downloaded successfully")

        except Exception as e:
            print(f"   ✗ Error downloading {model['name']}: {e}")
            logger.error(f"Failed to download {model['model_id']}: {e}")

    print("\n" + "=" * 60)
    print("✓ Model download complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SecureRAG models")
    parser.add_argument(
        "--models-dir",
        type=str,
        help="Directory to cache models (default: ~/.cache/huggingface)"
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else None

    download_models(models_dir)
