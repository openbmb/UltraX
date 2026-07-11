#!/usr/bin/env python3
"""
Batch data refinement using an optimized prompt.

Reads parquet files, refines each row's content via LLM API,
and writes results with matching filenames.

Supports:
  - Configurable concurrent file workers
  - Resume from breakpoint (skips already-completed output files)

Usage:
  python refine_dataset.py --input_dir <dir> --output_dir <dir> --prompt_file <file> --api_key <key>
"""

import os
import time
import logging
import argparse
import threading
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_lock = threading.Lock()
_stats = {"calls": 0, "tokens": 0, "errors": 0}


def call_api(system_prompt, user_content, api_url, api_key, model,
             temperature=0, max_tokens=24000, timeout=180, retry_times=3, retry_delay=2):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(retry_times):
        try:
            resp = requests.post(api_url, json=body, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            with _lock:
                _stats["calls"] += 1
                usage = data.get("usage", {})
                _stats["tokens"] += usage.get("total_tokens", 0)

            return content
        except Exception:
            if attempt < retry_times - 1:
                time.sleep(retry_delay * (attempt + 1))

    with _lock:
        _stats["errors"] += 1
    return None


def process_file(input_path, output_path, prompt, api_url, api_key, model):
    fname = os.path.basename(input_path)
    table = pq.read_table(input_path, columns=["content"])
    contents = table.column("content").to_pylist()

    originals, refined, success_flags = [], [], []
    for text in contents:
        result = call_api(prompt, text, api_url, api_key, model)
        originals.append(text)
        if result is not None:
            refined.append(result)
            success_flags.append("yes")
        else:
            refined.append("")
            success_flags.append("no")

    out_table = pa.table({
        "original_content": originals,
        "refined_content": refined,
        "refinement_success": success_flags,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    pq.write_table(out_table, tmp_path)
    os.rename(tmp_path, output_path)

    ok = sum(1 for f in success_flags if f == "yes")
    return fname, len(contents), ok


def run_dataset(ds_name, input_dir, output_dir, prompt_file, api_url, api_key, model, concurrency):
    ds_input = os.path.join(input_dir, ds_name)
    ds_output = os.path.join(output_dir, ds_name)

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    all_files = sorted(f for f in os.listdir(ds_input) if f.endswith(".parquet"))
    os.makedirs(ds_output, exist_ok=True)

    todo, skipped = [], 0
    for fname in all_files:
        output_path = os.path.join(ds_output, fname)
        if os.path.isfile(output_path):
            skipped += 1
        else:
            todo.append(fname)

    logger.info("[%s] Input: %s (%d files), To process: %d, Skipped: %d",
                ds_name, ds_input, len(all_files), len(todo), skipped)

    if not todo:
        return

    total_rows, total_ok, done = 0, 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for fname in todo:
            inp = os.path.join(ds_input, fname)
            out = os.path.join(ds_output, fname)
            futures[pool.submit(process_file, inp, out, prompt, api_url, api_key, model)] = fname

        for future in as_completed(futures):
            fname = futures[future]
            try:
                _, n, ok = future.result()
                total_rows += n
                total_ok += ok
                done += 1
                logger.info("[%s] [%d/%d] %s rows=%d ok=%d", ds_name, done, len(todo), fname, n, ok)
            except Exception as exc:
                done += 1
                logger.error("[%s] [%d/%d] %s FAILED: %s", ds_name, done, len(todo), fname, exc)

    logger.info("[%s] DONE files=%d rows=%d ok=%d elapsed=%.0fs",
                ds_name, len(all_files), total_rows, total_ok, time.time() - t0)


def main():
    parser = argparse.ArgumentParser(description="Batch LLM data refinement")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root input directory (contains dataset subdirectories with parquet files)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory")
    parser.add_argument("--prompt_dir", type=str, required=True,
                        help="Directory containing per-dataset optimized_prompt.txt files")
    parser.add_argument("--api_url", type=str, required=True,
                        help="LLM API endpoint URL")
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", ""),
                        help="API key (or set API_KEY env var)")
    parser.add_argument("--model", type=str, default="deepseek-v3",
                        help="Model name for API calls")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Number of concurrent file workers (default: 64)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset names (default: all subdirectories)")
    args = parser.parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(d for d in os.listdir(args.input_dir)
                          if os.path.isdir(os.path.join(args.input_dir, d)))

    logger.info("Datasets to process: %s", datasets)
    for ds in datasets:
        prompt_file = os.path.join(args.prompt_dir, ds, "optimized_prompt.txt")
        if not os.path.isfile(prompt_file):
            logger.warning("Prompt file not found for %s: %s, skipping", ds, prompt_file)
            continue
        run_dataset(ds, args.input_dir, args.output_dir, prompt_file,
                    args.api_url, args.api_key, args.model, args.concurrency)
    logger.info("All datasets complete.")


if __name__ == "__main__":
    main()
