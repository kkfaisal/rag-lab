"""Download public RAG benchmarks to your machine (they are not part of the repository).

Each one lands in data/datasets/<name>/ as documents + questions.jsonl, ready for
rag-lab's Datasets page. With --load it is also loaded into the document store, so
ingestion pipelines can use it right away.

    uv run python scripts/get_datasets.py --list
    uv run python scripts/get_datasets.py scifact nfcorpus --load
    uv run python scripts/get_datasets.py enterprise_rag_bench --load     # 1.3 GB download
    uv run python scripts/get_datasets.py msmarco --split dev              # another question split

Any BEIR dataset name on https://huggingface.co/BeIR works, not only the listed ones.
Each dataset has its own license; check it before you use or share the data.
"""

import argparse
import sys

from dotenv import load_dotenv

from rag_lab import benchmarks, paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public RAG benchmarks into data/datasets/.")
    parser.add_argument("names", nargs="*", help="benchmarks to fetch (see --list)")
    parser.add_argument("--list", action="store_true", help="show the available benchmarks")
    parser.add_argument("--load", action="store_true", help="also load them into the document store")
    parser.add_argument("--split", default="test", help="BEIR question split: test (default), dev or train")
    args = parser.parse_args()
    load_dotenv(paths.ROOT / ".env")

    if args.list or not args.names:
        print("Available benchmarks (name, download size, description):\n" + benchmarks.listing())
        if not args.names:
            print("\nExample: uv run python scripts/get_datasets.py scifact --load")
        return 0
    failed = []
    for name in args.names:
        print(f"\n== {name}")
        try:
            benchmarks.fetch(name, load=args.load, split=args.split)
        except (Exception, SystemExit) as e:          # keep going with the others
            print(f"   failed: {e}")
            failed.append(name)
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
