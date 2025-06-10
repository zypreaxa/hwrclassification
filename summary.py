"""
summarize_sections.py

Summarization pipeline for IAM-Words section texts.

This script:
 1. Loads a pretrained transformer summarization model (e.g. bart-large-cnn).
 2. Reads each .txt file in a given directory (grouped by section).
 3. Splits long texts into chunks if necessary.
 4. Generates summaries for each chunk and concatenates them.
 5. Writes per-section summary .txt files.
 6. Optionally aggregates all summaries into a CSV file.

Usage:
    python summary.py --text-dir data_output/text_sections --out-dir data_output/summaries --model facebook/bart-large-cnn --max-chunk-tokens 800 --min-length 50 --max-length 200 --aggregate-csv data/summaries.csv
"""
import argparse
import os
from pathlib import Path
import csv

from transformers import pipeline, AutoTokenizer
from tqdm import tqdm


def chunk_text(text, tokenizer, max_tokens: int):
    """
    Naively split text into chunks of at most max_tokens tokens using the tokenizer.
    Splits by sentences to avoid chopping in the middle of words.
    """
    # Split on sentences by simple period heuristic
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        tok_len = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_len + tok_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = tok_len
        else:
            current_chunk.append(sent)
            current_len += tok_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def run_summarization(
    text_dir: Path,
    out_dir: Path,
    model_name: str,
    max_chunk_tokens: int,
    min_length: int,
    max_length: int,
    aggregate_csv: Path = None
) -> int:
    os.makedirs(out_dir, exist_ok=True)

    summarizer = pipeline("summarization", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    section_files = sorted(text_dir.glob("*.txt"))
    aggregate_data = []

    # Section-level progress bar
    for path in tqdm(section_files, desc="Summarizing sections", unit="section"):
        section_id = path.stem
        text = path.read_text(encoding="utf-8")

        chunks = chunk_text(text, tokenizer, max_chunk_tokens)
        summary_chunks = []

        # Chunk-level progress bar (collapsed after finishing each section)
        for chunk in tqdm(chunks, desc=f"{section_id}", unit="chunk", leave=False):
            summary = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]["summary_text"]
            summary_chunks.append(summary)

        full_summary = " ".join(summary_chunks)

        # Write per-section summary
        out_path = out_dir / f"{section_id}_summary.txt"
        out_path.write_text(full_summary, encoding="utf-8")

        if aggregate_csv:
            aggregate_data.append((section_id, full_summary))

    # Write aggregate CSV if requested
    if aggregate_csv:
        with aggregate_csv.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["section", "summary"])
            for section_id, summary in aggregate_data:
                writer.writerow([section_id, summary])

    return len(section_files)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize each text section using a pretrained transformer model"
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        required=True,
        help="Directory containing section .txt files to summarize",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write summary .txt files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-cnn",
        help="Hugging Face model name for summarization",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=800,
        help="Maximum number of tokens per chunk",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum length of each summary",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum length of each summary",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        help="Optional CSV file to aggregate all summaries",
    )
    args = parser.parse_args()

    count = run_summarization(
        text_dir=args.text_dir,
        out_dir=args.out_dir,
        model_name=args.model,
        max_chunk_tokens=args.max_chunk_tokens,
        min_length=args.min_length,
        max_length=args.max_length,
        aggregate_csv=args.aggregate_csv,
    )
    print(f"Summaries generated for {count} sections in '{args.out_dir}'")
    if args.aggregate_csv:
        print(f"Aggregate CSV: {args.aggregate_csv}")


if __name__ == "__main__":
    main()
