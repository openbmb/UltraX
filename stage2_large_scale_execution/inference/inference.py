#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UltraX Large-Scale Inference Script (Data-Parallel, Function-Only Output)

This script performs large-scale batch inference using a trained refinement model
to generate cleaning operations (function calls) for pre-training data.

Features:
1. Reads parquet files from an input directory (expects 'original' column)
2. Applies sliding window segmentation for long documents
3. Uses vLLM for batch inference with data parallelism (one model per GPU)
4. Merges sliding window results with overlap-aware aggregation
5. Outputs model predictions without executing the actual cleaning

Output:
- One output parquet file per input file
- Two columns: original, model_output

Data Parallelism:
- Each GPU hosts a full model replica
- Files are distributed across GPUs in round-robin fashion
- Multiprocessing with one process per GPU

Usage:
    python inference.py --input_dir /path/to/data --model_path /path/to/model --output_dir /path/to/output
"""

import os
import re
import logging
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

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


def setup_logger(gpu_id: int = None):
    """Configure logger for the current process."""
    if gpu_id is not None:
        log_format = f'%(asctime)s - GPU{gpu_id} - %(levelname)s - %(message)s'
    else:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    return logging.getLogger(__name__)


# ==================== Configuration ====================
@dataclass
class Config:
    """Runtime configuration."""
    input_dir: str
    model_path: str
    output_dir: str
    max_chars: int
    overlap_ratio: float
    gpu_memory_utilization: float
    max_model_len: int
    max_new_tokens: int
    num_gpus: int
    infer_batch_size: int
    read_batch_size: int


# ==================== Function Parsing ====================

def parse_quoted_string(s: str, quote_char: str) -> Tuple[str, str]:
    """Parse a quoted string with escape character handling."""
    if not s.startswith(quote_char):
        raise ValueError(f"String does not start with {quote_char}")

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


def parse_function_call(func_str: str) -> Optional[Dict]:
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


def parse_function_list(content: str) -> List[Dict]:
    """Parse a newline-separated list of function calls."""
    if not content or not content.strip():
        return []

    functions = []
    for line in content.strip().split('\n'):
        line = line.strip()
        if line:
            func_info = parse_function_call(line)
            if func_info:
                functions.append(func_info)
    return functions


def extract_prediction(response: str) -> str:
    """Extract prediction from model output, stripping <think>...</think> prefix."""
    prefix = "<think>\n\n</think>\n\n"
    if response.startswith(prefix):
        return response[len(prefix):]
    return response


def functions_to_string(functions: List[Dict]) -> str:
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
            lines.append(f"remove_lines({func['start_line']}, {func['end_line']})")
        elif op == 'replace_str':
            lines.append(f"replace_str({func['line_number']}, {repr(func['search_content'])}, {repr(func['replace_content'])})")
        elif op == 'add_line':
            lines.append(f"add_line({func['base_position']}, {func['sub_index']}, {repr(func['content'])})")

    return '\n'.join(lines)


# ==================== Sliding Window ====================

def normalize_newlines(text: str) -> str:
    """Normalize \\r\\n and \\r to \\n."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def add_line_markers(content: str) -> str:
    """Add line number markers (<lid:N>) to each line."""
    lines = normalize_newlines(content).split('\n')
    marked_lines = []
    for i, line in enumerate(lines, start=1):
        marked_lines.append(f"<lid:{i}>{line}")
    return '\n'.join(marked_lines)


@dataclass
class WindowSegment:
    """A sliding window segment of a document."""
    content: str
    start_line: int
    end_line: int
    overlap_start: int
    is_first: bool = False


def split_by_sliding_window(
    content: str,
    max_chars: int,
    overlap_ratio: float = 0.2
) -> List[WindowSegment]:
    """Split content into sliding window segments based on character count."""
    lines = normalize_newlines(content).split('\n')
    total_lines = len(lines)

    if total_lines == 0:
        return []

    marked_content = add_line_markers(content)
    total_chars = len(marked_content)

    if total_chars <= max_chars:
        return [WindowSegment(
            content=marked_content,
            start_line=1,
            end_line=total_lines,
            overlap_start=1,
            is_first=True
        )]

    segments = []
    current_start = 0

    while current_start < total_lines:
        left = current_start
        right = total_lines
        best_end = current_start + 1

        while left < right:
            mid = (left + right + 1) // 2
            segment_lines = lines[current_start:mid]
            marked_segment = []
            for i, line in enumerate(segment_lines, start=current_start + 1):
                marked_segment.append(f"<lid:{i}>{line}")
            marked_content = '\n'.join(marked_segment)

            chars = len(marked_content)

            if chars <= max_chars:
                best_end = mid
                left = mid
            else:
                right = mid - 1

        segment_lines = lines[current_start:best_end]
        marked_segment = []
        for i, line in enumerate(segment_lines, start=current_start + 1):
            marked_segment.append(f"<lid:{i}>{line}")
        marked_content = '\n'.join(marked_segment)

        is_first = (current_start == 0)
        overlap_start = 1 if is_first else (current_start + 1)

        segments.append(WindowSegment(
            content=marked_content,
            start_line=current_start + 1,
            end_line=best_end,
            overlap_start=overlap_start,
            is_first=is_first
        ))

        current_window_chars = len(marked_content)
        overlap_chars = int(current_window_chars * overlap_ratio)

        overlap_start_line = best_end
        accumulated_chars = 0
        for line_idx in range(best_end - 1, current_start - 1, -1):
            line_with_marker = f"<lid:{line_idx + 1}>{lines[line_idx]}"
            line_chars = len(line_with_marker)
            if accumulated_chars + line_chars <= overlap_chars:
                accumulated_chars += line_chars
                overlap_start_line = line_idx
            else:
                break

        next_start = overlap_start_line
        if next_start <= current_start:
            next_start = best_end

        current_start = next_start

        if best_end >= total_lines:
            break

    for i in range(1, len(segments)):
        prev_end = segments[i-1].end_line
        segments[i].overlap_start = prev_end + 1

    return segments


def adjust_function_for_segment(
    func: Dict,
    segment: WindowSegment,
    is_segment_mode: bool = False
) -> Optional[Dict]:
    """Adjust function line numbers to only affect the non-overlap region."""
    operation = func.get('operation')

    if not is_segment_mode:
        return func

    non_overlap_start = segment.overlap_start
    segment_end = segment.end_line

    if operation == 'keep_all':
        return None

    if operation == 'remove_all':
        if non_overlap_start > segment_end:
            return None
        return {
            'operation': 'remove_lines',
            'start_line': non_overlap_start,
            'end_line': segment_end
        }

    if operation == 'remove_lines':
        start = func['start_line']
        end = func['end_line']

        if end < non_overlap_start:
            return None
        if start < non_overlap_start:
            start = non_overlap_start
        if end > segment_end:
            end = segment_end

        return {
            'operation': 'remove_lines',
            'start_line': start,
            'end_line': end
        }

    if operation == 'replace_str':
        line_num = func['line_number']
        if line_num < non_overlap_start or line_num > segment_end:
            return None
        return func.copy()

    if operation == 'add_line':
        base_pos = func['base_position']
        if base_pos < non_overlap_start or base_pos > segment_end:
            return None
        return func.copy()

    return func


def merge_window_functions(
    window_results: List[Tuple[WindowSegment, List[Dict]]],
    total_lines: int
) -> Tuple[List[Dict], bool, bool]:
    """Merge function results from all windows with overlap-aware aggregation."""
    if len(window_results) == 0:
        return [], True, False

    if len(window_results) == 1:
        segment, funcs = window_results[0]
        if len(funcs) == 1:
            if funcs[0]['operation'] == 'keep_all':
                return [], True, False
            if funcs[0]['operation'] == 'remove_all':
                return [], False, True
        return funcs, False, False

    is_segment_mode = True
    merged_functions = []

    for segment, funcs in window_results:
        for func in funcs:
            adjusted = adjust_function_for_segment(func, segment, is_segment_mode)
            if adjusted:
                merged_functions.append(adjusted)

    if not merged_functions:
        return [], True, False

    all_remove_lines = all(f['operation'] == 'remove_lines' for f in merged_functions)
    if all_remove_lines:
        covered_lines = set()
        for f in merged_functions:
            for ln in range(f['start_line'], f['end_line'] + 1):
                covered_lines.add(ln)
        if covered_lines == set(range(1, total_lines + 1)):
            return [], False, True

    return merged_functions, False, False


# ==================== Data I/O ====================

class PerFileParquetWriter:
    """Writes output parquet files (columns: original, model_output) per input file."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_buffers: Dict[str, List[Dict]] = defaultdict(list)

    def add(self, filename: str, original: str, model_output: str):
        """Add a record to the buffer."""
        self.file_buffers[filename].append({
            'original': original,
            'model_output': model_output
        })

    def flush_file(self, filename: str):
        """Flush the buffer for a specific file to disk."""
        if filename not in self.file_buffers or not self.file_buffers[filename]:
            return

        df = pd.DataFrame(self.file_buffers[filename])
        output_file = self.output_dir / filename
        df.to_parquet(output_file, index=False)

        self.file_buffers[filename] = []

    def close(self):
        """Flush all remaining buffers to disk."""
        for filename, records in self.file_buffers.items():
            if not records:
                continue
            df = pd.DataFrame(records)
            output_file = self.output_dir / filename
            df.to_parquet(output_file, index=False)

        self.file_buffers.clear()


# ==================== Binary Batch Inference ====================

def binary_batch_inference(
    llm,
    sampling_params,
    prompts: List[str],
    mappings: List[Tuple[int, int]],
    tasks: List,
    logger,
    min_batch_size: int = 1
) -> Tuple[int, int]:
    """
    Batch inference with binary split on failure.

    When a batch fails, it is split in half and each half is retried recursively.
    Prompts that still fail at min_batch_size are marked as keep_all().

    Returns:
        (success_count, failed_count)
    """
    if not prompts:
        return 0, 0

    try:
        outputs = llm.generate(prompts, sampling_params)

        for (task_idx, seg_idx), output in zip(mappings, outputs):
            response = output.outputs[0].text
            task = tasks[task_idx]
            while len(task.segment_responses) <= seg_idx:
                task.segment_responses.append("")
            task.segment_responses[seg_idx] = response

        return len(prompts), 0

    except Exception as e:
        batch_size = len(prompts)

        if batch_size <= min_batch_size:
            failed_count = 0
            for task_idx, seg_idx in mappings:
                task = tasks[task_idx]
                task.failed_segments.add(seg_idx)
                while len(task.segment_responses) <= seg_idx:
                    task.segment_responses.append("")
                task.segment_responses[seg_idx] = "keep_all()"
                failed_count += 1
            logger.warning(f"Batch inference failed (size={batch_size}), marking as keep_all: {e}")
            return 0, failed_count

        mid = batch_size // 2
        logger.warning(f"Batch inference failed (size={batch_size}), splitting and retrying: {e}")

        left_success, left_failed = binary_batch_inference(
            llm, sampling_params,
            prompts[:mid], mappings[:mid],
            tasks, logger, min_batch_size
        )

        right_success, right_failed = binary_batch_inference(
            llm, sampling_params,
            prompts[mid:], mappings[mid:],
            tasks, logger, min_batch_size
        )

        return left_success + right_success, left_failed + right_failed


# ==================== Single-GPU Worker Process ====================

@dataclass
class InferTask:
    """Inference task for a single document."""
    data_idx: int
    filename: str
    content: str
    segments: List[WindowSegment] = field(default_factory=list)
    segment_responses: List[str] = field(default_factory=list)
    failed_segments: set = field(default_factory=set)


def worker_process(
    gpu_id: int,
    config: Config,
    file_list: List[str],
    result_queue: mp.Queue
):
    """
    Single-GPU worker process.

    Args:
        gpu_id: GPU ID to bind to
        config: Runtime configuration
        file_list: List of filenames assigned to this GPU
        result_queue: Queue for reporting results back to main process
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger = setup_logger(gpu_id)

    logger.info(f"Worker started, assigned {len(file_list)} files")

    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'split_count': 0,
        'keep_all_count': 0,
        'remove_all_count': 0,
        'modified_count': 0
    }

    if not file_list:
        logger.info("No files to process")
        result_queue.put(('done', gpu_id, stats))
        return

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )

        logger.info("Loading vLLM model...")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=config.model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            tensor_parallel_size=1,
            enforce_eager=False
        )

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=-1
        )

        logger.info("Model loaded successfully")

        output_writer = PerFileParquetWriter(config.output_dir)

        input_path = Path(config.input_dir)

        for file_idx, filename in enumerate(file_list):
            logger.info(f"Processing file [{file_idx + 1}/{len(file_list)}]: {filename}")

            file_path = input_path / filename
            parquet_file = pq.ParquetFile(file_path)

            total_rows = parquet_file.metadata.num_rows
            logger.info(f"Total rows in file: {total_rows}")

            file_processed = 0
            batch_idx = 0

            for record_batch in parquet_file.iter_batches(batch_size=config.read_batch_size, columns=['original']):
                batch_idx += 1
                batch_df = record_batch.to_pandas()

                if 'original' not in batch_df.columns:
                    if batch_idx == 1:
                        logger.warning(f"File {filename} missing 'original' column, skipping")
                    break

                contents = batch_df['original'].tolist()
                batch_size = len(contents)
                stats['total'] += batch_size
                file_processed += batch_size

                logger.info(f"  Batch {batch_idx}: processing {batch_size} rows ({file_processed}/{total_rows})")

                tasks = []
                for idx, content in enumerate(contents):
                    task = InferTask(
                        data_idx=idx,
                        filename=filename,
                        content=content if content else ""
                    )

                    if content and content.strip():
                        segments = split_by_sliding_window(
                            content,
                            config.max_chars,
                            config.overlap_ratio
                        )
                        task.segments = segments

                        if len(segments) > 1:
                            stats['split_count'] += 1

                    tasks.append(task)

                all_prompts = []
                prompt_to_task_segment = []

                for task_idx, task in enumerate(tasks):
                    for seg_idx, seg in enumerate(task.segments):
                        messages = [
                            {"role": "system", "content": SYSTEM_INSTRUCTION},
                            {"role": "user", "content": seg.content},
                        ]
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        all_prompts.append(prompt)
                        prompt_to_task_segment.append((task_idx, seg_idx))

                if all_prompts:
                    infer_start_time = time.time()
                    total_infer_prompts = 0
                    skipped_prompts = 0

                    for batch_start in range(0, len(all_prompts), config.infer_batch_size):
                        batch_end = min(batch_start + config.infer_batch_size, len(all_prompts))
                        batch_prompts = all_prompts[batch_start:batch_end]
                        batch_mapping = prompt_to_task_segment[batch_start:batch_end]

                        success, failed = binary_batch_inference(
                            llm, sampling_params,
                            batch_prompts, batch_mapping,
                            tasks, logger, min_batch_size=1
                        )
                        total_infer_prompts += success
                        skipped_prompts += failed

                    infer_time = time.time() - infer_start_time
                    throughput = total_infer_prompts / infer_time if infer_time > 0 else 0
                    logger.info(f"    Inference done: {total_infer_prompts} prompts, {skipped_prompts} failed, {infer_time:.2f}s, {throughput:.1f} prompts/s")

                for task in tasks:
                    try:
                        content = task.content

                        if not content or not content.strip():
                            output_writer.add(task.filename, content if content else "", "")
                            stats['failed'] += 1
                            continue

                        if not task.segments:
                            output_writer.add(task.filename, content, "")
                            stats['failed'] += 1
                            continue

                        if len(task.failed_segments) == len(task.segments):
                            output_writer.add(task.filename, content, "# all segments failed")
                            stats['failed'] += 1
                            logger.warning(f"All segments failed, keeping original: {task.filename}, data_idx={task.data_idx}")
                            continue

                        while len(task.segment_responses) < len(task.segments):
                            task.segment_responses.append("keep_all()")

                        pred_strs = [extract_prediction(r) for r in task.segment_responses]

                        window_results = []
                        for seg_idx, (seg, pred_str) in enumerate(zip(task.segments, pred_strs)):
                            funcs = parse_function_list(pred_str)
                            window_results.append((seg, funcs))

                        total_lines = len(normalize_newlines(content).split('\n'))
                        merged_funcs, is_keep_all, is_remove_all = merge_window_functions(
                            window_results, total_lines
                        )

                        if is_keep_all:
                            model_output_str = "keep_all()"
                            stats['keep_all_count'] += 1
                        elif is_remove_all:
                            model_output_str = "remove_all()"
                            stats['remove_all_count'] += 1
                        else:
                            model_output_str = functions_to_string(merged_funcs)
                            stats['modified_count'] += 1
                        if task.failed_segments:
                            model_output_str += f"\n# Failed segments: {sorted(task.failed_segments)}"

                        output_writer.add(task.filename, content, model_output_str)
                        stats['success'] += 1

                    except Exception as e:
                        output_writer.add(task.filename, task.content if task.content else "", f"# ERROR: {str(e)}")
                        stats['failed'] += 1

                del tasks
                del all_prompts
                del prompt_to_task_segment

            output_writer.flush_file(filename)

            logger.info(f"File completed: {filename}, rows processed: {file_processed}")

        output_writer.close()

        logger.info("=" * 40)
        logger.info(f"GPU {gpu_id} processing complete")
        logger.info(f"Total rows: {stats['total']}")
        logger.info(f"Success: {stats['success']}, Failed: {stats['failed']}")
        logger.info(f"keep_all: {stats['keep_all_count']}, remove_all: {stats['remove_all_count']}, modified: {stats['modified_count']}")
        logger.info("=" * 40)

    except Exception as e:
        logger.error(f"GPU {gpu_id} encountered an error: {e}")
        import traceback
        traceback.print_exc()

    result_queue.put(('done', gpu_id, stats))


# ==================== Main ====================

def get_files_to_process(config: Config) -> Tuple[List[str], int]:
    """Get list of files to process, skipping already-completed ones."""
    input_path = Path(config.input_dir)
    output_path = Path(config.output_dir)

    parquet_files = sorted(input_path.glob("*.parquet"))

    files_to_process = []
    skipped = 0

    for pf in parquet_files:
        output_file = output_path / pf.name
        if output_file.exists():
            skipped += 1
            continue
        files_to_process.append(pf.name)

    return files_to_process, skipped


def main():
    parser = argparse.ArgumentParser(
        description="UltraX large-scale inference: generate cleaning operations via data-parallel vLLM."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing parquet files with 'original' column")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained refinement model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for parquet files (columns: original, model_output)")
    parser.add_argument("--max_chars", type=int, default=48000,
                        help="Max characters per sliding window segment (default: 48000)")
    parser.add_argument("--overlap_ratio", type=float, default=0.2,
                        help="Sliding window overlap ratio (default: 0.2)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="vLLM GPU memory utilization (default: 0.85)")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="vLLM max model length (default: 32768)")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max new tokens to generate (default: 4096)")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs for data parallelism (default: 8)")
    parser.add_argument("--infer_batch_size", type=int, default=1000,
                        help="Inference batch size per GPU (default: 1000)")
    parser.add_argument("--read_batch_size", type=int, default=500,
                        help="Streaming read batch size from parquet (default: 500)")
    args = parser.parse_args()

    logger = setup_logger()

    config = Config(
        input_dir=args.input_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_chars=args.max_chars,
        overlap_ratio=args.overlap_ratio,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        num_gpus=args.num_gpus,
        infer_batch_size=args.infer_batch_size,
        read_batch_size=args.read_batch_size
    )

    if not Path(config.input_dir).exists():
        logger.error(f"Input directory does not exist: {config.input_dir}")
        return

    logger.info("Scanning files...")
    files_to_process, skipped = get_files_to_process(config)

    logger.info(f"Total files: {len(files_to_process) + skipped}")
    if skipped > 0:
        logger.info(f"Skipped (already processed): {skipped}")
    logger.info(f"Files to process: {len(files_to_process)}")

    if not files_to_process:
        logger.info("All files have been processed")
        return

    files_per_gpu = [[] for _ in range(config.num_gpus)]
    for i, filename in enumerate(files_to_process):
        gpu_idx = i % config.num_gpus
        files_per_gpu[gpu_idx].append(filename)

    logger.info("=" * 60)
    logger.info("Data Parallel Configuration")
    logger.info("=" * 60)
    for gpu_id in range(config.num_gpus):
        logger.info(f"GPU {gpu_id}: assigned {len(files_per_gpu[gpu_id])} files")
    logger.info("=" * 60)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting worker processes...")

    result_queue = mp.Queue()
    processes = []

    for gpu_id in range(config.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, config, files_per_gpu[gpu_id], result_queue)
        )
        p.start()
        processes.append(p)

    total_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'split_count': 0,
        'keep_all_count': 0,
        'remove_all_count': 0,
        'modified_count': 0
    }

    completed = 0
    while completed < config.num_gpus:
        result = result_queue.get()
        if result[0] == 'done':
            _, gpu_id, stats = result
            logger.info(f"GPU {gpu_id} completed")
            for key in total_stats:
                total_stats[key] += stats.get(key, 0)
            completed += 1

    for p in processes:
        p.join()

    logger.info("=" * 60)
    logger.info("All processing complete - Summary")
    logger.info("=" * 60)
    logger.info(f"Total rows:      {total_stats['total']}")
    logger.info(f"Success:         {total_stats['success']}")
    logger.info(f"Failed:          {total_stats['failed']}")
    logger.info(f"Split documents: {total_stats['split_count']}")
    logger.info(f"keep_all:        {total_stats['keep_all_count']}")
    logger.info(f"remove_all:      {total_stats['remove_all_count']}")
    logger.info(f"Modified:        {total_stats['modified_count']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
