#!/usr/bin/env python3
"""
export_text_groups.py

Standalone script to read IAM Words dataset index (data/words.txt) and
group word transcriptions by the first three characters of the word ID,
writing one .txt file per group in the order words.txt provides.

Usage:
    python export_text_groups.py --words data/words.txt --out-dir data_output/text_sections [--include-err]
"""
import os
import argparse
from collections import defaultdict

def export_text_groups(words_txt: str, out_dir: str, include_err: bool = False) -> int:
    """
    Parse the IAM words index and write one .txt per group (first 3 chars of ID).

    Args:
        words_txt: Path to the IAM words.txt file.
        out_dir: Directory where group .txt files will be saved.
        include_err: If True, include lines marked 'err'; otherwise only 'ok'.

    Returns:
        Number of group files exported.
    """
    groups = defaultdict(list)

    # Read in order, grouping by first 3 chars of word_id
    with open(words_txt, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            # Skip malformed lines
            if len(parts) < 9:
                continue
            status = parts[1]
            if not include_err and status != 'ok':
                continue

            word_id = parts[0]
            transcription = parts[-1]
            group_id = word_id[:3]

            groups[group_id].append(transcription)

    os.makedirs(out_dir, exist_ok=True)
    for group_id, words in groups.items():
        full_text = " ".join(words)
        outfile = os.path.join(out_dir, f"{group_id}.txt")
        with open(outfile, "w", encoding="utf-8") as out_f:
            out_f.write(full_text)

    return len(groups)


def main():
    parser = argparse.ArgumentParser(
        description="Export IAM Words text by 3-char groups to files."
    )
    parser.add_argument(
        "--words", default="data/words.txt",
        help="Path to the IAM words.txt index file"
    )
    parser.add_argument(
        "--out-dir", default="data_output/text_sections",
        help="Output directory for group .txt files"
    )
    parser.add_argument(
        "--include-err", action="store_true",
        help="Include lines marked 'err' in addition to 'ok'"
    )
    args = parser.parse_args()

    count = export_text_groups(
        words_txt=args.words,
        out_dir=args.out_dir,
        include_err=args.include_err
    )
    print(f"Exported {count} group files to '{args.out_dir}'")

if __name__ == "__main__":
    main()
