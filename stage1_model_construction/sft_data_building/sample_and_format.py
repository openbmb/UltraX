#!/usr/bin/env python3
"""
Sample and format training data for SFT.

Samples data from each function combo category according to target counts
(with oversampling for underrepresented combos), adds system instructions,
and writes shuffled output as JSONL shards.
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

LINES_PER_FILE = 1000

SKIP_DATASETS = {"by_combo"}

SYSTEM_INSTRUCTION = """\
You are a web text cleaner for LLM pre-training data. Clean the given web-crawled text by removing noise while maximally preserving valuable content.

Input: each line is prefixed with <lid:N> (1-indexed line number).

## Operations
- keep_all() — Text is already clean, no changes needed
- remove_all() — Entire document is valueless (e.g., error pages, login walls, garbled text)
- remove_lines(start, end) — Remove lines start to end inclusive
- replace_str(line_number, 'old', 'new') — Replace substring within a line (for inline noise removal, line merging, or fixing HTML entities)
- add_line(base_position, sub_index, 'content') — Insert a new line near base_position

## Rules
1. REMOVE: navigation/breadcrumbs, ads/banners, copyright/cookie notices, "Share"/"Subscribe"/"Sign up" prompts, SEO stuffing, boilerplate templates, e-commerce UI elements (prices, "Add to cart", stock status), "Related posts"/"You may also like" sections, and comment section headers
2. PRESERVE: article body, factual content, quotes, data, author info with context, dates in narrative
3. When in doubt, KEEP the content — over-deletion is worse than under-deletion
4. Prefer remove_lines when entire lines are noise; use replace_str only for inline noise within otherwise valuable lines
5. Use remove_all() only when the document has absolutely zero informational value
6. Output operations only, one per line"""

# ============================================================
# Sampling configuration
# ============================================================

COMBO_TARGETS = {
    "remove_lines_replace_str": 731340,
    "remove_lines":             231148,
    "replace_str":              146613,
    "remove_all":               82544,
    "add_line_remove_lines_replace_str": 6861,
    "add_line_remove_lines":    4588,
    "add_line_replace_str":     1075,
    "add_line":                 40,
}

KEEP_ALL_TARGET = 1806314


def oversample(data: list, target: int) -> list:
    if len(data) == 0:
        return []
    if len(data) >= target:
        return random.sample(data, target)
    result = list(data)
    remaining = target - len(data)
    result.extend(random.choices(data, k=remaining))
    return result


def add_instruction(raw_json_str: str) -> str:
    """Add system instruction to a JSON data entry, returning the new JSON string."""
    data = json.loads(raw_json_str)
    old_messages = data["messages"]
    data["messages"] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        *old_messages,
    ]
    return json.dumps(data, ensure_ascii=False)


def load_all_data_by_combo(train_data_dir: Path):
    combo_data = defaultdict(list)
    keep_all_data = []

    dataset_dirs = [
        d for d in train_data_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DATASETS
    ]

    print(f"  Scanning {len(dataset_dirs)} dataset directories...")

    for ds_dir in sorted(dataset_dirs):
        ds_name = ds_dir.name
        file_count = 0
        ds_keep_all = 0
        ds_other = 0
        for jsonl_file in ds_dir.rglob("*.jsonl"):
            file_count += 1
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        combo = data.pop("combo", "unknown")
                        clean_line = json.dumps(data, ensure_ascii=False)

                        if combo == "keep_all":
                            keep_all_data.append(clean_line)
                            ds_keep_all += 1
                        else:
                            combo_data[combo].append(clean_line)
                            ds_other += 1
                    except Exception:
                        pass
        print(f"    {ds_name}: {file_count} files, keep_all={ds_keep_all}, other={ds_other}")

    return dict(combo_data), keep_all_data


def write_output(all_lines: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(all_lines)

    total = len(all_lines)
    file_idx = 0
    for start in range(0, total, LINES_PER_FILE):
        batch = all_lines[start:start + LINES_PER_FILE]
        output_file = output_dir / f"train_{file_idx:05d}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in batch:
                f.write(line + "\n")
        file_idx += 1

    print(f"\nOutput complete:")
    print(f"  Total entries: {total}")
    print(f"  Number of files: {file_idx}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample training data by combo category and add system instructions."
    )
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Directory containing per-file JSONL training data "
                             "(output of function_construction.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write sampled and formatted output JSONL shards")
    args = parser.parse_args()

    train_data_dir = Path(args.train_data_dir)
    output_dir = Path(args.output_dir)

    # 1. Load data
    print("=" * 60)
    print("Step 1: Load data")
    print("=" * 60)

    combo_data, keep_all_data = load_all_data_by_combo(train_data_dir)

    print(f"\n  Data counts per combo:")
    for combo in sorted(combo_data.keys()):
        print(f"    {combo}: {len(combo_data[combo])}")
    print(f"    keep_all: {len(keep_all_data)}")

    # 2. Sample non-keep_all categories
    all_sampled = []

    print()
    print("=" * 60)
    print("Step 2: Sample non-keep_all categories")
    print("=" * 60)

    for combo_name, target in COMBO_TARGETS.items():
        data = combo_data.get(combo_name, [])
        sampled = oversample(data, target)
        all_sampled.extend(sampled)
        ratio = target / len(data) if data else 0
        print(f"  {combo_name}: {len(data)} -> {len(sampled)} ({ratio:.2f}x)")

    non_keep_all_total = len(all_sampled)

    # 3. Oversample keep_all to target count
    print()
    print("=" * 60)
    print("Step 3: Oversample keep_all")
    print("=" * 60)

    print(f"  Original keep_all data: {len(keep_all_data)} entries")
    print(f"  Target count: {KEEP_ALL_TARGET} entries")

    keep_all_sampled = oversample(keep_all_data, KEEP_ALL_TARGET)
    all_sampled.extend(keep_all_sampled)

    print(f"  Sampled: {len(keep_all_sampled)} entries (oversampled {len(keep_all_sampled)/len(keep_all_data):.2f}x)")

    # 4. Add system instruction
    print()
    print("=" * 60)
    print("Step 4: Add system instruction")
    print("=" * 60)

    all_sampled = [add_instruction(line) for line in all_sampled]
    print(f"  Added instruction to {len(all_sampled)} entries")

    # 5. Write output
    print()
    print("=" * 60)
    print("Step 5: Shuffle and write output files")
    print("=" * 60)

    write_output(all_sampled, output_dir)

    # 6. Statistics
    print()
    print("=" * 60)
    print("Final statistics")
    print("=" * 60)
    total = len(all_sampled)

    combo_counts = {"keep_all": len(keep_all_sampled)}
    for combo_name, target in COMBO_TARGETS.items():
        combo_counts[combo_name] = target

    print(f"\n  {'combo':<45s} {'count':>10s}  {'ratio':>7s}")
    print(f"  {'-'*65}")
    for combo_name in sorted(combo_counts.keys()):
        cnt = combo_counts[combo_name]
        print(f"  {combo_name:<45s} {cnt:>10,}  ({cnt/total*100:5.2f}%)")
    print(f"  {'-'*65}")
    print(f"  {'Total':<45s} {total:>10,}  (100.00%)")
    print(f"\n  2-epoch total training volume: {total * 2:,}")


if __name__ == "__main__":
    main()
