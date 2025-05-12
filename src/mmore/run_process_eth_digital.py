import os
import re
import gzip
import json
import argparse
import time
import yaml
import click
from typing import List
from multiprocessing import freeze_support

from src.mmore.process.processors.pdf_processor import PDFProcessor
from src.mmore.process.processors.base import ProcessorConfig
from src.mmore.type import MultimodalSample

CHUNK_SIZE = 1000


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def to_jsonl_gz(file_path: str, samples: List["MultimodalSample"]) -> None:
    with gzip.open(file_path, "at", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


def collect_processed_pdfs(output_dir: str) -> set:
    """Collect all processed file paths from existing .jsonl.gz archives."""
    processed = set()
    for fname in os.listdir(output_dir):
        if fname.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(output_dir, fname), "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        file_path = obj.get("metadata", {}).get("file_path")
                        if file_path:
                            processed.add(file_path)
                    except json.JSONDecodeError:
                        continue
    return processed


def get_next_part_index(output_dir: str) -> int:
    """Get the next part index based on existing part_XXXXX.jsonl.gz files."""
    max_index = -1
    pattern = re.compile(r"part_(\d{5})\.jsonl\.gz")
    for fname in os.listdir(output_dir):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            max_index = max(max_index, idx)
    return max_index + 1


def process(config_file: str):
    """Process documents from a directory, skipping already processed PDFs."""
    click.echo(f'Dispatcher configuration file path: {config_file}')
    overall_start_time = time.time()

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_pdfs = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    ])

    click.echo(f"Found {len(all_pdfs)} PDF files in {input_dir}")
    processed_pdfs = collect_processed_pdfs(output_dir)
    click.echo(f"Found {len(processed_pdfs)} already processed PDFs")

    pdf_file_paths = [fp for fp in all_pdfs if fp not in processed_pdfs]
    click.echo(f"{len(pdf_file_paths)} PDFs remaining to process")

    if not pdf_file_paths:
        click.echo("Nothing to do. All PDFs are already processed.")
        return

    processor_config = ProcessorConfig(custom_config={"output_path": output_dir})
    processor = PDFProcessor(config=processor_config)

    start_idx = get_next_part_index(output_dir)
    for i, chunk in enumerate(chunk_list(pdf_file_paths, CHUNK_SIZE)):
        part_idx = start_idx + i
        click.echo(f"Processing chunk {part_idx} with {len(chunk)} files...")
        results = processor.process_batch(chunk, fast_mode=True, num_workers=4)

        part_file = os.path.join(output_dir, f"part_{part_idx:05}.jsonl.gz")
        to_jsonl_gz(part_file, results)
        click.echo(f"Saved chunk {part_idx} to {part_file}")

    overall_end_time = time.time()
    click.echo(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds")


if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Process PDFs using MMORE and save in .jsonl.gz chunks.")
    parser.add_argument("--config_file", required=True, help="YAML config file with input_dir and output_dir.")
    args = parser.parse_args()

    process(args.config_file)
