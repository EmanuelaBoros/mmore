import os
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


def process(config_file: str):
    """Process documents from a directory."""
    click.echo(f'Dispatcher configuration file path: {config_file}')
    overall_start_time = time.time()

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    pdf_file_paths = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    ])

    click.echo(f"Found {len(pdf_file_paths)} PDF files in {input_dir}")

    processor_config = ProcessorConfig(custom_config={"output_path": output_dir})
    processor = PDFProcessor(config=processor_config)

    for i, chunk in enumerate(chunk_list(pdf_file_paths, CHUNK_SIZE)):
        click.echo(f"Processing chunk {i} with {len(chunk)} files...")
        results = processor.process_batch(chunk, fast=True, num_workers=4)

        part_file = os.path.join(output_dir, f"part_{i:05}.jsonl.gz")
        to_jsonl_gz(part_file, results)
        click.echo(f"Saved chunk {i} to {part_file}")

    overall_end_time = time.time()
    click.echo(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds")


if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Process PDFs using MMORE and save in .jsonl.gz chunks.")
    parser.add_argument("--config_file", required=True, help="YAML config file with input_dir and output_dir.")
    args = parser.parse_args()

    process(args.config_file)
