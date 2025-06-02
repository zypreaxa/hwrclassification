"""
ner_pipeline.py

Named Entity Recognition pipeline for IAM-Words section texts.

This script:
 1. Loads a spaCy transformer-based NER model.
 2. Reads each .txt file in a given directory (grouped by section).
 3. Performs NER to extract entities.
 4. Generates HTML visualizations via displaCy.
 5. Writes per-section JSON files of entities.
 6. Aggregates all entities into a summary CSV.

Usage:
    python NER.py --text-dir data_output/text_sections --viz-dir data_output/visualizations --out-dir data_output/ner_results --summary-csv data_output/entity_summary.csv --model en_core_web_trf --batch-size 8
"""
import argparse
import json
import csv
from pathlib import Path

import spacy
from spacy import displacy

def run_ner_pipeline(text_dir: Path,
                     viz_dir: Path,
                     out_dir: Path,
                     # summary_csv: Path,
                     model: str = "en_core_web_trf",
                     batch_size: int = 8):
    # Load model
    print(f"Loading spaCy model '{model}'...")
    nlp = spacy.load(model)
    # Prepare directories
    text_paths = sorted(text_dir.glob("*.txt"))
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read texts and process in batches
    sections = [p.stem for p in text_paths]
    texts = [p.read_text(encoding="utf-8") for p in text_paths]

    print(f"Processing {len(texts)} sections with batch size {batch_size}...")
    for section_id, doc in zip(sections, nlp.pipe(texts, batch_size=batch_size)):
        # 1. HTML visualization
        html = displacy.render(doc, style="ent", page=True)
        (viz_dir / f"{section_id}.html").write_text(html, encoding="utf-8")

        # 2. JSON entities
        entities = [{
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        } for ent in doc.ents]
        (out_dir / f"{section_id}.json").write_text(
            json.dumps(entities, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


    """
    # Prepare CSV writer
    with summary_csv.open("w", newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["section", "entity_text", "label"])  # header

        # Read texts and process in batches
        sections = [p.stem for p in text_paths]
        texts = [p.read_text(encoding="utf-8") for p in text_paths]

        print(f"Processing {len(texts)} sections with batch size {batch_size}...")
        for section_id, doc in zip(sections, nlp.pipe(texts, batch_size=batch_size)):
            # 1. HTML visualization
            html = displacy.render(doc, style="ent", page=True)
            (viz_dir / f"{section_id}.html").write_text(html, encoding="utf-8")

            # 2. JSON entities
            entities = [{
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            } for ent in doc.ents]
            (out_dir / f"{section_id}.json").write_text(
                json.dumps(entities, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            # 3. Append to summary CSV
            for ent in doc.ents:
                csv_writer.writerow([section_id, ent.text, ent.label_])
    """

    print("NER pipeline completed.")
    print(f"HTML visualizations: {viz_dir}/")
    print(f"JSON outputs:       {out_dir}/")
    # print(f"Summary CSV:       {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="NER pipeline for IAM-Words sections")
    parser.add_argument("--text-dir", type=Path, default=Path("data/text_sections"),
                        help="Directory containing section .txt files")
    parser.add_argument("--viz-dir", type=Path, default=Path("data/visualizations"),
                        help="Directory for HTML visualizations")
    parser.add_argument("--out-dir", type=Path, default=Path("data/ner_results"),
                        help="Directory for per-section JSON outputs")
    parser.add_argument("--summary-csv", type=Path, default=Path("data/entity_summary.csv"),
                        help="CSV file summarizing all entities")
    parser.add_argument("--model", type=str, default="en_core_web_trf",
                        help="spaCy model name to use for NER")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for spaCy nlp.pipe")
    args = parser.parse_args()

    run_ner_pipeline(
        text_dir=args.text_dir,
        viz_dir=args.viz_dir,
        out_dir=args.out_dir,
        summary_csv=args.summary_csv,
        model=args.model,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
