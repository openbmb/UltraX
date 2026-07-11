#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sliding window splitter for long documents (multi-process version).
Splits documents exceeding max_tokens into overlapping windows at line boundaries.
Uses a configurable tokenizer for token counting.
"""

import os
import math
import argparse
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Tuple, Dict
from multiprocessing import Pool
from transformers import AutoTokenizer

_tokenizer = None
_max_tokens = None
_step_tokens = None


def _init_worker(tokenizer_path: str, max_tokens: int, overlap_ratio: float):
    global _tokenizer, _max_tokens, _step_tokens
    _tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, model_max_length=10**8
    )
    _max_tokens = max_tokens
    _step_tokens = int(max_tokens * (1 - overlap_ratio))


def _count_tokens(text: str) -> int:
    if pd.isna(text) or text == "":
        return 0
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _normalize_newlines(text: str) -> str:
    return text.replace('\r\n', '\n').replace('\r', '\n')


def _split_content(content: str) -> List[str]:
    content = _normalize_newlines(content)
    total_tokens = _count_tokens(content)
    if total_tokens <= _max_tokens:
        return [content]

    lines = content.split('\n')
    chunks = []
    start_idx = 0

    while start_idx < len(lines):
        current_chunk_lines = []
        current_tokens = 0
        idx = start_idx

        while idx < len(lines):
            line = lines[idx]
            line_tokens = _count_tokens(line + '\n')
            if current_tokens + line_tokens <= _max_tokens:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
                idx += 1
            else:
                if not current_chunk_lines:
                    current_chunk_lines.append(line)
                    current_tokens += line_tokens
                    idx += 1
                break

        if current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))

        if idx >= len(lines):
            break

        accumulated_tokens = 0
        next_start_idx = start_idx
        while next_start_idx < idx and accumulated_tokens < _step_tokens:
            line_tokens = _count_tokens(lines[next_start_idx] + '\n')
            accumulated_tokens += line_tokens
            next_start_idx += 1

        if next_start_idx == start_idx:
            next_start_idx += 1
        start_idx = next_start_idx

    return chunks if chunks else [content]


def _process_single_file(args: Tuple[str, str]) -> Dict:
    input_path, output_path = args
    input_path, output_path = Path(input_path), Path(output_path)

    stats = {
        "input_file": input_path.name,
        "dataset": input_path.parent.name,
        "total_rows": 0,
        "rows_under_limit": 0,
        "rows_split": 0,
        "output_chunks": 0,
        "total_tokens_after_split": 0,
        "error": None,
    }

    try:
        parquet_file = pq.ParquetFile(input_path)
        if 'content' not in parquet_file.schema.names:
            stats["error"] = "missing content column"
            return stats

        all_chunks = []
        for batch in parquet_file.iter_batches(batch_size=500, columns=['content']):
            df_batch = batch.to_pandas()
            for content in df_batch['content']:
                stats["total_rows"] += 1
                if pd.isna(content) or not content.strip():
                    continue

                token_count = _count_tokens(content)
                if token_count <= _max_tokens:
                    stats["rows_under_limit"] += 1
                    stats["output_chunks"] += 1
                    normalized = _normalize_newlines(content)
                    all_chunks.append(normalized)
                    stats["total_tokens_after_split"] += token_count
                else:
                    stats["rows_split"] += 1
                    chunks = _split_content(content)
                    stats["output_chunks"] += len(chunks)
                    all_chunks.extend(chunks)
                    for c in chunks:
                        stats["total_tokens_after_split"] += _count_tokens(c)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if all_chunks:
            df = pd.DataFrame({"content": all_chunks})
            df.to_parquet(output_path, engine='pyarrow', index=False)

    except Exception as e:
        stats["error"] = str(e)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Sliding window splitter for long documents")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for split parquet files")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer (e.g., Qwen3-1.7B)")
    parser.add_argument("--max_tokens", type=int, default=12000, help="Maximum tokens per window (default: 12000)")
    parser.add_argument("--overlap_ratio", type=float, default=0.2, help="Overlap ratio between windows (default: 0.2)")
    args = parser.parse_args()

    source_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    tokenizer_path = args.tokenizer_path
    max_tokens = args.max_tokens
    overlap_ratio = args.overlap_ratio

    cpu_count = os.cpu_count() or 4
    num_workers = max(1, math.floor(cpu_count * 0.8))

    print("=" * 80)
    print("Sliding Window Document Splitter")
    print(f"Source directory : {source_root}")
    print(f"Output directory : {output_root}")
    print(f"Max tokens       : {max_tokens}")
    print(f"Overlap ratio    : {overlap_ratio:.0%}")
    print(f"Step tokens      : {int(max_tokens * (1 - overlap_ratio))}")
    print(f"CPU cores        : {cpu_count}")
    print(f"Workers          : {num_workers}")
    print("=" * 80)

    tasks: List[Tuple[str, str]] = []
    skipped = 0
    datasets = sorted([d for d in source_root.iterdir() if d.is_dir()])

    for dataset_path in datasets:
        dataset_name = dataset_path.name
        parquet_files = sorted([f for f in dataset_path.iterdir() if f.suffix == '.parquet'])
        for pf in parquet_files:
            out_file = output_root / dataset_name / pf.name
            if out_file.exists():
                skipped += 1
            else:
                tasks.append((str(pf), str(out_file)))

    print(f"\nFound {len(datasets)} datasets")
    print(f"  To process: {len(tasks)} files")
    print(f"  Skipped   : {skipped} files (output exists)\n")

    if not tasks:
        print("No new files to process")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting {num_workers} workers...\n")
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_path, max_tokens, overlap_ratio),
    ) as pool:
        results = []
        for stats in pool.imap_unordered(_process_single_file, tasks):
            results.append(stats)
            tag = f"[{stats['dataset']}/{stats['input_file']}]"
            if stats["error"]:
                print(f"  x {tag} Error: {stats['error']}")
            else:
                print(f"  + {tag}  {stats['total_rows']} rows -> "
                      f"{stats['output_chunks']} chunks, "
                      f"tokens={stats['total_tokens_after_split']:,}")

    total_rows = sum(r["total_rows"] for r in results)
    total_under = sum(r["rows_under_limit"] for r in results)
    total_split = sum(r["rows_split"] for r in results)
    total_chunks = sum(r["output_chunks"] for r in results)
    total_tokens = sum(r["total_tokens_after_split"] for r in results)
    total_errors = sum(1 for r in results if r["error"])

    ds_stats: Dict[str, Dict] = {}
    for r in results:
        ds = r["dataset"]
        if ds not in ds_stats:
            ds_stats[ds] = {"rows": 0, "chunks": 0, "tokens": 0, "files": 0}
        ds_stats[ds]["rows"] += r["total_rows"]
        ds_stats[ds]["chunks"] += r["output_chunks"]
        ds_stats[ds]["tokens"] += r["total_tokens_after_split"]
        ds_stats[ds]["files"] += 1

    print("\n" + "=" * 80)
    print("Per-dataset statistics:")
    print("-" * 80)
    for ds_name in sorted(ds_stats):
        s = ds_stats[ds_name]
        print(f"  {ds_name:30s}  files={s['files']:>4d}  "
              f"rows={s['rows']:>8,}  chunks={s['chunks']:>8,}  "
              f"tokens={s['tokens']:>14,}")

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Files processed : {len(results)}")
    print(f"  Errors          : {total_errors}")
    print(f"  Total rows      : {total_rows:,}")
    print(f"  Rows not split  : {total_under:,}")
    print(f"  Rows split      : {total_split:,}")
    print(f"  Output chunks   : {total_chunks:,}")
    print(f"  Total tokens    : {total_tokens:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
