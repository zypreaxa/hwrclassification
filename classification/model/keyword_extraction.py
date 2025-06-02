#!/usr/bin/env python3
"""
extract_keywords.py

Keyword extraction pipeline for IAM‐Words section texts using KeyBERT.

Steps:
 1. Load a KeyBERT model (sentence‐transformer embeddings).
 2. Read each .txt file in --text-dir.
 3. Extract top-N keywords (unigrams/bigrams by default).
 4. Write per‐section JSON with [(keyword, score), …].
 5. Optionally aggregate into a CSV (--aggregate-csv).

Usage example:
  python keyword_extraction.py --text-dir data_output/text_sections --out-dir data_output/keywords --embedding-model all-MiniLM-L6-v2 --keyphrase-ngram-range 1 2 --top-n 10 --use-mmr --diversity 0.5 --nr-candidates 20 --aggregate-csv data/keywords_summary.csv
"""
import argparse
from pathlib import Path
import json
import csv

from keybert import KeyBERT
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Extract keywords from texts using KeyBERT"
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        required=True,
        help="Directory containing section .txt files to process",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write per-section JSON keyword files",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for embeddings",
    )
    parser.add_argument(
        "--keyphrase-ngram-range",
        type=int,
        nargs=2,
        metavar=("MIN_N", "MAX_N"),
        default=(1, 2),
        help="n-gram range for candidate keyphrases",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top keywords to extract per document",
    )
    parser.add_argument(
        "--use-mmr",
        action="store_true",
        help="Use MMR (Maximal Marginal Relevance) for diversity",
    )
    parser.add_argument(
        "--diversity",
        type=float,
        default=0.5,
        help="Diversity parameter for MMR (0 ≤ diversity ≤ 1)",
    )
    parser.add_argument(
        "--nr-candidates",
        type=int,
        default=20,
        help="Number of candidate phrases to score before selecting top-n",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        help="Optional CSV file to aggregate all keywords (section, keyword, score)",
    )
    args = parser.parse_args()

    # Prepare output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Gather section files
    section_files = sorted(args.text_dir.glob("*.txt"))
    if not section_files:
        print(f"No .txt files found in {args.text_dir!r}")
        return

    # Initialize KeyBERT
    kw_model = KeyBERT(model=args.embedding_model)

    # Container for aggregated CSV rows
    aggregate_rows = []

    # Loop with progress bar
    for path in tqdm(section_files, desc="Extracting keywords", unit="section"):
        section_id = path.stem
        text = path.read_text(encoding="utf-8")

        # Build extraction kwargs
        extract_kwargs = {
            "keyphrase_ngram_range": tuple(args.keyphrase_ngram_range),
            "stop_words": "english",
            "top_n": args.top_n,
            "nr_candidates": args.nr_candidates,
        }
        if args.use_mmr:
            extract_kwargs["use_mmr"] = True
            extract_kwargs["diversity"] = args.diversity
        else:
            extract_kwargs["use_maxsum"] = True

        # Extract keywords
        keywords = kw_model.extract_keywords(text, **extract_kwargs)
        # keywords: List of (keyword, score)

        # Write per-section JSON
        out_json = args.out_dir / f"{section_id}_keywords.json"
        out_json.write_text(
            json.dumps(
                [{"keyword": k, "score": s} for k, s in keywords],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Collect for CSV
        if args.aggregate_csv:
            for kw, score in keywords:
                aggregate_rows.append((section_id, kw, score))

    # Write aggregated CSV if requested
    if args.aggregate_csv:
        with args.aggregate_csv.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["section", "keyword", "score"])
            writer.writerows(aggregate_rows)

    print(f"\n✅ Keywords extracted for {len(section_files)} sections.")
    if args.aggregate_csv:
        print(f"Aggregated CSV written to {args.aggregate_csv!r}")


if __name__ == "__main__":
    main()
