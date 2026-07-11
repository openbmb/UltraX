#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UltraX Post-Processing and Execution Script

This script reads inference output (original + model_output columns), parses the
predicted function calls, applies post-processing (ambiguous filtering, same-line
merging, duplicate pattern detection), and executes the deterministic program to
produce cleaned text.

Features:
1. Reads parquet files with 'original' and 'model_output' columns
2. Parses and validates function calls from model_output
3. Filters ambiguous replace_str operations (multiple occurrences in target line)
4. Merges multiple replace_str operations on the same line
5. Detects repetitive/looping patterns via similarity clustering
6. Executes validated operations deterministically
7. Outputs 3 columns: original, cleaned, processed_functions

Usage:
    python post_process_and_execute.py --input_dir /path/to/inference_output --output_dir /path/to/cleaned
"""

import os
import re
import logging
import difflib
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq


def normalize_newlines(text: str) -> str:
    """Normalize \\r\\n and \\r to \\n.

    Must be consistent with the inference script's normalization,
    since line numbers are computed based on \\n-split lines.
    """
    if not isinstance(text, str):
        return text
    return text.replace('\r\n', '\n').replace('\r', '\n')


def setup_logger(worker_id=None):
    """Configure logger for the current process."""
    if worker_id is not None:
        log_format = '%(asctime)s - Worker{} - %(levelname)s - %(message)s'.format(worker_id)
    else:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    return logging.getLogger(__name__)


# ==================== Similarity Clustering ====================

def _find_similar_groups(strings, ratio_threshold):
    """Greedy clustering of strings by similarity ratio. Returns list of index groups."""
    groups = []
    for idx, s in enumerate(strings):
        merged = False
        for group in groups:
            rep = group[0]
            if difflib.SequenceMatcher(None, rep, s).ratio() >= ratio_threshold:
                group[1].append(idx)
                merged = True
                break
        if not merged:
            groups.append((s, [idx]))
    return [g[1] for g in groups]


# ==================== Repetitive Pattern Detection ====================

def detect_repetitive_functions(
    functions,
    add_line_exact_threshold=2,
    add_line_diff_pos_threshold=3,
    replace_str_content_threshold=30,
    remove_lines_threshold=3,
    add_line_similar_threshold=3,
    similarity_ratio=0.85,
    original=None,
    replace_str_legit_ratio=0.8,
):
    """
    Detect repetitive/looping patterns in function lists.

    Detection rules:
    1. add_line: identical (base_position, content) > exact_threshold
    2. add_line: same content but different positions > diff_pos_threshold
    3. replace_str: identical (search, replace) > replace_threshold
       Exemption: if search appears in original >= count * legit_ratio times,
       it's treated as legitimate high-frequency template cleaning.
    4. remove_lines: identical (start, end) > remove_threshold
    5. add_line: content similarity >= ratio in cluster > similar_threshold
    """
    min_threshold = min(add_line_exact_threshold, remove_lines_threshold)
    if len(functions) < min_threshold + 1:
        return False

    add_line_exact_counts = defaultdict(int)
    replace_str_content_counts = defaultdict(int)
    remove_lines_counts = defaultdict(int)
    add_line_by_content = defaultdict(set)
    add_line_contents = []

    for func in functions:
        op = func.get('operation')
        if op == 'add_line':
            base_pos = func.get('base_position')
            content = func.get('content', '')
            add_line_exact_counts[(base_pos, content)] += 1
            add_line_by_content[content].add(base_pos)
            add_line_contents.append(content)
        elif op == 'replace_str':
            key = (func.get('search_content'), func.get('replace_content'))
            replace_str_content_counts[key] += 1
        elif op == 'remove_lines':
            key = (func.get('start_line'), func.get('end_line'))
            remove_lines_counts[key] += 1

    for count in add_line_exact_counts.values():
        if count > add_line_exact_threshold:
            return True

    for positions in add_line_by_content.values():
        if len(positions) > add_line_diff_pos_threshold:
            return True

    for (search, _replace), count in replace_str_content_counts.items():
        if count > replace_str_content_threshold:
            if original and isinstance(search, str) and search:
                occ = original.count(search)
                if occ >= count * replace_str_legit_ratio:
                    continue
            return True

    for count in remove_lines_counts.values():
        if count > remove_lines_threshold:
            return True

    if len(add_line_contents) > add_line_similar_threshold:
        groups = _find_similar_groups(add_line_contents, similarity_ratio)
        for group_indices in groups:
            if len(group_indices) > add_line_similar_threshold:
                return True

    return False


# ==================== Function Parsing ====================

def parse_quoted_string(s, quote_char):
    """Parse a quoted string with escape character handling."""
    if not s.startswith(quote_char):
        raise ValueError("Expected quote char")

    result = []
    i = 1
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char == 'n':
                result.append('\n')
            elif next_char == 't':
                result.append('\t')
            elif next_char == 'r':
                result.append('\r')
            elif next_char == '\\':
                result.append('\\')
            elif next_char == quote_char:
                result.append(quote_char)
            else:
                result.append('\\')
                result.append(next_char)
            i += 2
        elif s[i] == quote_char:
            return ''.join(result), s[i + 1:]
        else:
            result.append(s[i])
            i += 1

    raise ValueError("String not properly terminated")


def parse_function_call(func_str):
    """Parse a function call string into a structured dict."""
    func_str = func_str.strip()
    if not func_str:
        return None

    if func_str == 'keep_all()':
        return {'operation': 'keep_all'}

    if func_str == 'remove_all()':
        return {'operation': 'remove_all'}

    match = re.match(r'remove_lines\((\d+),\s*(\d+)\)', func_str)
    if match:
        return {
            'operation': 'remove_lines',
            'start_line': int(match.group(1)),
            'end_line': int(match.group(2))
        }

    if func_str.startswith('replace_str('):
        try:
            inner = func_str[len('replace_str('):-1]
            first_comma = inner.find(',')
            if first_comma == -1:
                return None

            line_num = int(inner[:first_comma].strip())
            rest = inner[first_comma + 1:].strip()

            if rest.startswith("'"):
                search_content, rest = parse_quoted_string(rest, "'")
            elif rest.startswith('"'):
                search_content, rest = parse_quoted_string(rest, '"')
            else:
                return None

            rest = rest.lstrip(',').strip()

            if rest.startswith("'"):
                replace_content, _ = parse_quoted_string(rest, "'")
            elif rest.startswith('"'):
                replace_content, _ = parse_quoted_string(rest, '"')
            else:
                return None

            return {
                'operation': 'replace_str',
                'line_number': line_num,
                'search_content': search_content,
                'replace_content': replace_content
            }
        except Exception:
            return None

    if func_str.startswith('add_line('):
        try:
            inner = func_str[len('add_line('):-1]
            first_comma = inner.find(',')
            if first_comma == -1:
                return None

            base_pos = int(inner[:first_comma].strip())
            rest = inner[first_comma + 1:].strip()

            second_comma = rest.find(',')
            if second_comma == -1:
                return None

            sub_index = int(rest[:second_comma].strip())
            rest = rest[second_comma + 1:].strip()

            if rest.startswith("'"):
                content, _ = parse_quoted_string(rest, "'")
            elif rest.startswith('"'):
                content, _ = parse_quoted_string(rest, '"')
            else:
                return None

            return {
                'operation': 'add_line',
                'base_position': base_pos,
                'sub_index': sub_index,
                'content': content
            }
        except Exception:
            return None

    return None


def parse_function_list(content):
    """Parse a newline-separated list of function calls, skipping comments and separators."""
    if not content or not content.strip():
        return []

    functions = []
    for line in content.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and line != '---':
            func_info = parse_function_call(line)
            if func_info:
                functions.append(func_info)
    return functions


def extract_prediction(response):
    """Strip <think>...</think> blocks from model output."""
    if not response:
        return ""
    return re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()


def functions_to_string(functions):
    """Convert a list of function dicts back to string representation."""
    if not functions:
        return ""

    lines = []
    for func in functions:
        op = func.get('operation', '')
        if op == 'keep_all':
            lines.append('keep_all()')
        elif op == 'remove_all':
            lines.append('remove_all()')
        elif op == 'remove_lines':
            lines.append("remove_lines({}, {})".format(func['start_line'], func['end_line']))
        elif op == 'replace_str':
            lines.append("replace_str({}, {}, {})".format(
                func['line_number'], repr(func['search_content']), repr(func['replace_content'])))
        elif op == 'add_line':
            lines.append("add_line({}, {}, {})".format(
                func['base_position'], func['sub_index'], repr(func['content'])))

    return '\n'.join(lines)


# ==================== Function Post-Processing ====================

def extract_modifications(search, replace, pos_in_line):
    """Extract actual modifications from search/replace content using diff."""
    matcher = difflib.SequenceMatcher(None, search, replace)
    modifications = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        abs_start = pos_in_line + i1
        abs_end = pos_in_line + i2
        new_content = replace[j1:j2]
        modifications.append((abs_start, abs_end, new_content))

    return modifications


def force_merge_all_ops(ops, orig_line, line_num):
    """Force-merge all replace_str operations on a single line into one operation."""
    if not ops:
        return ops

    all_modifications = []
    processed_mod_ranges = []

    def trim_modification(start, end, content):
        current_start, current_end = start, end

        for p_start, p_end in processed_mod_ranges:
            if current_end <= p_start or current_start >= p_end:
                continue
            if p_start <= current_start and current_end <= p_end:
                return None
            if p_start <= current_start < p_end < current_end:
                trim_len = p_end - current_start
                current_start = p_end
                content = content[trim_len:] if trim_len < len(content) else ''
            elif current_start < p_start < current_end <= p_end:
                trim_len = current_end - p_start
                current_end = p_start
                content = content[:-trim_len] if trim_len < len(content) else ''
            elif current_start < p_start and p_end < current_end:
                return None

        if current_start >= current_end:
            return None

        return (current_start, current_end, content)

    sorted_ops = sorted(ops, key=lambda op: len(op['search_content']), reverse=True)
    search_next_start = {}

    for op in sorted_ops:
        search = op['search_content']
        replace = op['replace_content']
        start_from = search_next_start.get(search, 0)

        while True:
            pos = orig_line.find(search, start_from)
            if pos == -1:
                break

            mods = extract_modifications(search, replace, pos)

            if not mods:
                start_from = pos + 1
                continue

            trimmed_mods = []
            for mod in mods:
                s, e, c = mod
                trimmed = trim_modification(s, e, c)
                if trimmed:
                    trimmed_mods.append(trimmed)

            if trimmed_mods:
                all_modifications.extend(trimmed_mods)
                for s, e, _ in trimmed_mods:
                    processed_mod_ranges.append((s, e))

            search_next_start[search] = pos + 1
            break

    if not all_modifications:
        return ops

    all_modifications.sort(key=lambda x: x[0])

    merged_modifications = []
    for start, end, content in all_modifications:
        if merged_modifications:
            last_start, last_end, last_content = merged_modifications[-1]
            if start <= last_end:
                new_end = max(last_end, end)
                overlap_in_last = max(0, start - last_start)
                new_content = last_content[:overlap_in_last] + content
                merged_modifications[-1] = (last_start, new_end, new_content)
                continue
        merged_modifications.append((start, end, content))

    if not merged_modifications:
        return ops

    result_line = orig_line
    for start, end, new_content in reversed(merged_modifications):
        result_line = result_line[:start] + new_content + result_line[end:]

    if result_line == orig_line:
        return ops

    return [{
        'operation': 'replace_str',
        'line_number': line_num,
        'search_content': orig_line,
        'replace_content': result_line
    }]


def merge_all_same_line_replace_ops(operations, original_lines):
    """Filter ambiguous operations and force-merge same-line replace_str operations."""
    replace_ops = [op for op in operations if op['operation'] == 'replace_str']
    other_ops = [op for op in operations if op['operation'] != 'replace_str']

    if not replace_ops:
        return operations

    # Filter: search_content must appear exactly once in the target line
    filtered_ops = []
    for op in replace_ops:
        line_num = op['line_number']
        if isinstance(line_num, int) and 1 <= line_num <= len(original_lines):
            orig_line = original_lines[line_num - 1]
            if orig_line.count(op['search_content']) == 1:
                filtered_ops.append(op)
        else:
            filtered_ops.append(op)
    replace_ops = filtered_ops

    line_ops = defaultdict(list)
    for op in replace_ops:
        line_ops[op['line_number']].append(op)

    merged_ops = []

    for line_num in sorted(line_ops.keys()):
        ops = line_ops[line_num]

        if len(ops) == 1:
            merged_ops.append(ops[0])
            continue

        if isinstance(line_num, int) and 1 <= line_num <= len(original_lines):
            orig_line = original_lines[line_num - 1]
        else:
            merged_ops.extend(ops)
            continue

        merged = force_merge_all_ops(ops, orig_line, line_num)
        merged_ops.extend(merged)

    return merged_ops + other_ops


# ==================== Clean Function Generation & Execution ====================

def generate_clean_function_code(operations):
    """Generate executable clean_data function code from operation list."""
    code_lines = [
        "import re",
        "from typing import Dict",
        "",
        "def remove_lines(line_dict, start_line, end_line):",
        "    return {k: v for k, v in line_dict.items() if not (start_line <= k <= end_line)}",
        "",
        "def remove_all():",
        "    return ''",
        "",
        "def add_line(line_dict, base_pos, sub_index, content):",
        "    result = line_dict.copy()",
        "    position = base_pos + sub_index / 10000.0",
        "    result[position] = content",
        "    return result",
        "",
        "def replace_str(line_dict, line_number, old_content, new_content):",
        "    result = line_dict.copy()",
        "    if line_number in result:",
        "        line = result[line_number]",
        "        if old_content in line:",
        "            result[line_number] = line.replace(old_content, new_content, 1)",
        "    return result",
        "",
        "def keep_all(content):",
        "    return content",
        "",
        "def clean_data(content):",
        "    lines = content.splitlines(keepends=True)",
        "    line_dict = {i: line for i, line in enumerate(lines, start=1)}",
        ""
    ]

    if not operations:
        code_lines.append("    return keep_all(content)")
    else:
        if len(operations) == 1 and operations[0]['operation'] == 'keep_all':
            code_lines.append("    return keep_all(content)")
        elif len(operations) == 1 and operations[0]['operation'] == 'remove_all':
            code_lines.append("    return remove_all()")
        else:
            for op in operations:
                if op['operation'] == 'remove_lines':
                    code_lines.append("    line_dict = remove_lines(line_dict, {}, {})".format(
                        op['start_line'], op['end_line']))
                elif op['operation'] == 'add_line':
                    code_lines.append("    line_dict = add_line(line_dict, {}, {}, {})".format(
                        op['base_position'], op['sub_index'], repr(op['content'])))
                elif op['operation'] == 'replace_str':
                    code_lines.append("    line_dict = replace_str(line_dict, {}, {}, {})".format(
                        op['line_number'], repr(op['search_content']), repr(op['replace_content'])))

            code_lines.append("")
            code_lines.append("    sorted_keys = sorted(line_dict.keys())")
            code_lines.append("    sorted_lines = []")
            code_lines.append("    for _i, _k in enumerate(sorted_keys):")
            code_lines.append("        _line = line_dict[_k]")
            code_lines.append("        if isinstance(_k, float):")
            code_lines.append("            if sorted_lines and sorted_lines[-1] and not sorted_lines[-1].endswith('\\n'):")
            code_lines.append("                sorted_lines[-1] = sorted_lines[-1] + '\\n'")
            code_lines.append("            if _i + 1 < len(sorted_keys) and _line and not _line.endswith('\\n'):")
            code_lines.append("                _line = _line + '\\n'")
            code_lines.append("        sorted_lines.append(_line)")
            code_lines.append("    result = ''.join(sorted_lines)")
            code_lines.append("    result = result.replace('\\r\\n', '\\n').replace('\\r', '\\n')")
            code_lines.append("    result = re.sub(r'\\n\\n+', '\\n', result)")
            code_lines.append("    result = result.strip('\\n')")
            code_lines.append("    return result")

    code_lines.append("")
    return "\n".join(code_lines)


def execute_clean_function(clean_func_code, original_content):
    """Execute the generated clean function in an isolated namespace."""
    try:
        namespace = {}
        exec(clean_func_code, namespace)
        if 'clean_data' in namespace:
            return namespace['clean_data'](original_content)
    except Exception:
        pass
    return None


# ==================== Row Processing ====================

def process_row(original, model_output):
    """
    Process a single row: parse model_output -> post-process -> execute cleaning.

    Returns:
        (original, cleaned, processed_functions)
    """
    if not isinstance(original, str):
        original = ""

    original = normalize_newlines(original)

    if not original.strip():
        return (original, original, "keep_all()")

    mo_str = model_output if (model_output and isinstance(model_output, str)) else ""
    raw_functions = parse_function_list(mo_str)

    original_lines = original.splitlines(keepends=True)
    functions = list(raw_functions)
    if functions:
        functions = merge_all_same_line_replace_ops(functions, original_lines)

    if detect_repetitive_functions(raw_functions, original=original):
        processed_str = functions_to_string(functions) + "\n# FILTERED: repetitive pattern detected"
        return (original, original, processed_str)

    if not functions or (len(functions) == 1 and functions[0]['operation'] == 'keep_all'):
        return (original, original, "keep_all()")

    if len(functions) == 1 and functions[0]['operation'] == 'remove_all':
        return (original, "", "remove_all()")

    processed_str = functions_to_string(functions)

    clean_code = generate_clean_function_code(functions)
    cleaned = execute_clean_function(clean_code, original)
    if cleaned is None:
        cleaned = original
        processed_str += "\n# FAILED: execution error"

    return (original, cleaned, processed_str)


# ==================== Worker Process ====================

def worker_process(worker_id, file_list, input_dir, output_dir, result_queue):
    """Worker process: processes assigned parquet files."""
    logger = setup_logger(worker_id)
    logger.info("Started, assigned {} files".format(len(file_list)))

    stats = {
        'total': 0,
        'keep_all': 0,
        'remove_all': 0,
        'modified': 0,
        'repetitive_filtered': 0,
        'failed': 0,
    }

    if not file_list:
        result_queue.put(('done', worker_id, stats))
        return

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for file_idx, filename in enumerate(file_list):
        file_start = time.time()
        logger.info("[{}/{}] Processing: {}".format(file_idx + 1, len(file_list), filename))

        try:
            df = pd.read_parquet(input_path / filename)
        except Exception as e:
            logger.error("Failed to read file {}: {}".format(filename, e))
            continue

        if 'original' not in df.columns or 'model_output' not in df.columns:
            logger.warning("File missing required columns, skipping: {}".format(filename))
            continue

        results = []
        for _, row in df.iterrows():
            orig = row.get('original', '')
            mo = row.get('model_output', '')
            result = process_row(orig, mo)
            results.append(result)

            stats['total'] += 1
            orig_normalized, cleaned_val, pf_val = result
            if '# FILTERED: repetitive' in pf_val:
                stats['repetitive_filtered'] += 1
                stats['keep_all'] += 1
            elif pf_val == 'keep_all()':
                stats['keep_all'] += 1
            elif pf_val == 'remove_all()':
                stats['remove_all'] += 1
            elif '# FAILED' in pf_val:
                stats['failed'] += 1
            else:
                cleaned_str = cleaned_val if isinstance(cleaned_val, str) else ""
                if cleaned_str == orig_normalized:
                    stats['keep_all'] += 1
                elif not cleaned_str.strip():
                    stats['remove_all'] += 1
                else:
                    stats['modified'] += 1

        out_df = pd.DataFrame(results, columns=[
            'original', 'cleaned', 'processed_functions'
        ])
        out_df.to_parquet(output_path / filename, index=False)

        elapsed = time.time() - file_start
        logger.info("  Done {}: {} rows, {:.1f}s".format(filename, len(results), elapsed))

    logger.info(
        "Worker {} complete | total={}, modified={}, keep_all={}, "
        "remove_all={}, repetitive={}, failed={}".format(
            worker_id, stats['total'], stats['modified'], stats['keep_all'],
            stats['remove_all'], stats['repetitive_filtered'], stats['failed'])
    )
    result_queue.put(('done', worker_id, stats))


# ==================== Main ====================

def get_files_to_process(input_dir, output_dir):
    """Get list of files to process, skipping already-completed ones."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    parquet_files = sorted(input_path.glob("*.parquet"))

    files_to_process = []
    skipped = 0
    for pf in parquet_files:
        if (output_path / pf.name).exists():
            skipped += 1
            continue
        files_to_process.append(pf.name)
    return files_to_process, skipped


def main():
    parser = argparse.ArgumentParser(
        description="UltraX post-processing and execution: parse, validate, and execute cleaning operations."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing inference output parquet files (columns: original, model_output)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for cleaned parquet files (3 columns: original, cleaned, processed_functions)")
    parser.add_argument("--num_workers", type=int, default=128,
                        help="Number of parallel worker processes (default: 128)")
    parser.add_argument("--similarity_ratio", type=float, default=0.85,
                        help="Similarity ratio threshold for duplicate detection (default: 0.85)")
    parser.add_argument("--add_line_similar_threshold", type=int, default=3,
                        help="Threshold for similar add_line pattern detection (default: 3)")
    args = parser.parse_args()

    logger = setup_logger()

    logger.info("Input directory:  {}".format(args.input_dir))
    logger.info("Output directory: {}".format(args.output_dir))
    logger.info("Workers:          {}".format(args.num_workers))
    logger.info("Similarity ratio: {}".format(args.similarity_ratio))
    logger.info("Add-line similar threshold: {}".format(args.add_line_similar_threshold))

    if not Path(args.input_dir).exists():
        logger.error("Input directory does not exist: {}".format(args.input_dir))
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Scanning files...")
    files_to_process, skipped = get_files_to_process(args.input_dir, args.output_dir)
    logger.info("Total files: {}, skipped: {}, to process: {}".format(
        len(files_to_process) + skipped, skipped, len(files_to_process)))

    if not files_to_process:
        logger.info("All files have been processed")
        return

    actual_workers = min(args.num_workers, len(files_to_process))
    files_per_worker = [[] for _ in range(actual_workers)]
    for i, filename in enumerate(files_to_process):
        files_per_worker[i % actual_workers].append(filename)

    logger.info("Using {} worker processes".format(actual_workers))
    for wid in range(actual_workers):
        logger.info("  Worker {}: {} files".format(wid, len(files_per_worker[wid])))

    start_time = time.time()
    result_queue = mp.Queue()
    processes = []

    for wid in range(actual_workers):
        p = mp.Process(
            target=worker_process,
            args=(wid, files_per_worker[wid], args.input_dir, args.output_dir, result_queue)
        )
        p.start()
        processes.append(p)

    total_stats = defaultdict(int)
    completed = 0
    while completed < actual_workers:
        result = result_queue.get()
        if result[0] == 'done':
            _, wid, sts = result
            for key, val in sts.items():
                total_stats[key] += val
            completed += 1

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("All processing complete")
    logger.info("=" * 60)
    logger.info("Total rows:          {:,}".format(total_stats['total']))
    logger.info("Modified:            {:,}".format(total_stats['modified']))
    logger.info("keep_all:            {:,}".format(total_stats['keep_all']))
    logger.info("remove_all:          {:,}".format(total_stats['remove_all']))
    logger.info("Repetitive filtered: {:,}".format(total_stats['repetitive_filtered']))
    logger.info("Execution failed:    {:,}".format(total_stats['failed']))
    logger.info("Elapsed time:        {:.1f}s ({:.1f}min)".format(elapsed, elapsed / 60))
    logger.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
