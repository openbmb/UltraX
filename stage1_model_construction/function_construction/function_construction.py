#!/usr/bin/env python3
"""
LAM + DCR Function Construction Algorithm
==========================================

This module implements the core algorithm for Stage 1 of UltraX:
- **LAM (Line-Aware Matching):** Matches lines between original and refined text
  using a global optimal matching strategy with content similarity, context
  similarity, and positional factors.
- **DCR (Diff-based Cleaning Representation):** Converts text-level differences
  into structured, fine-grained cleaning operations (replace_str, remove_lines,
  add_line, remove_all, keep_all).

The pipeline processes parquet files containing (original_content, refined_content)
pairs and outputs JSONL training data where each example maps line-id-marked
original text to a sequence of cleaning function calls.
"""

import argparse
import difflib
import json
import logging
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchDataCleaner:

    NON_MEANINGFUL_CHARS = set(
        # Whitespace characters (including fullwidth space)
        ' \u3000\u00a0\u2002\u2003\u2009'
        # English punctuation
        ',.;:!?\'\"'
        # Chinese punctuation
        '，。；：！？、''""'
        # Fullwidth English punctuation
        '．，；：！？＇＂'
        # Small punctuation variants
        '﹑﹐﹔﹕﹖﹗'
    )

    def __init__(self):
        pass

    def _strip_non_meaningful(self, s: str) -> str:
        """Remove non-meaningful characters, preserving substantive text content."""
        return ''.join(c for c in s if c not in self.NON_MEANINGFUL_CHARS)

    def is_meaningful_replace_str(self, op: Dict) -> bool:
        """Check whether a replace_str operation has substantive text changes.

        Returns False if the change only involves punctuation/whitespace.
        Returns True if there is a meaningful text content change.
        """
        if op['operation'] != 'replace_str':
            return True

        search = op['search_content']
        replace = op['replace_content']

        stripped_search = self._strip_non_meaningful(search)
        stripped_replace = self._strip_non_meaningful(replace)

        return stripped_search != stripped_replace

    def filter_meaningful_operations(self, operations: List[Dict]) -> List[Dict]:
        """Filter operation list, keeping only replace_str operations with substantive changes."""
        filtered = []

        for op in operations:
            if op['operation'] == 'replace_str':
                if self.is_meaningful_replace_str(op):
                    filtered.append(op)
            else:
                filtered.append(op)

        return filtered

    def make_context_unique_in_line(self, line: str, start: int, end: int) -> tuple:
        """Dynamically adjust context within a line to ensure target string uniqueness."""
        target = line[start:end]
        max_possible_len = max(start, len(line) - end)

        for context_len in range(0, max_possible_len + 1):
            before = line[max(0, start - context_len):start]
            after = line[end:min(len(line), end + context_len)]
            context_pattern = before + target + after
            if line.count(context_pattern) == 1:
                return before, target, after, True

        return line[0:start], target, line[end:], True

    def _get_line_similarity(self, line1: str, line2: str) -> float:
        """Compute similarity between two lines."""
        s1, s2 = line1.strip(), line2.strip()
        if s1 == s2:
            return 1.0
        matcher = difflib.SequenceMatcher(None, s1, s2)
        return matcher.ratio()

    def _get_context_similarity(self, orig_idx: int, ref_idx: int,
                                  orig_lines: List[Tuple[int, str]],
                                  ref_lines: List[Tuple[int, str]],
                                  context_size: int = 3) -> float:
        """Compute context similarity (context_size lines before and after)."""
        context_scores = []

        for i in range(1, context_size + 1):
            orig_prev = orig_idx - i
            ref_prev = ref_idx - i
            if orig_prev >= 0 and ref_prev >= 0:
                sim = self._get_line_similarity(
                    orig_lines[orig_prev][1],
                    ref_lines[ref_prev][1]
                )
                context_scores.append(sim)

        for i in range(1, context_size + 1):
            orig_next = orig_idx + i
            ref_next = ref_idx + i
            if orig_next < len(orig_lines) and ref_next < len(ref_lines):
                sim = self._get_line_similarity(
                    orig_lines[orig_next][1],
                    ref_lines[ref_next][1]
                )
                context_scores.append(sim)

        # Return neutral 0.5 when no context is available to avoid unfairly penalizing boundary lines
        return sum(context_scores) / len(context_scores) if context_scores else 0.5

    def match_lines(self, original_lines: List[str], refined_lines: List[str]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match original and refined lines using global optimal matching with context and position.

        Strategy:
        1. Compute content similarity, context similarity, and position similarity for all line pairs
        2. Combined score = content * 0.6 + context * 0.2 + position * 0.2
        3. Greedily match by descending combined score while preventing crossings

        Returns:
            matched_pairs: [(original_line_num, refined_line_num), ...]
            unmatched_original: [unmatched original line numbers, ...]
            unmatched_refined: [unmatched refined line numbers, ...]
        """
        orig_non_empty = [(i + 1, line) for i, line in enumerate(original_lines) if line.strip()]
        ref_non_empty = [(j + 1, line) for j, line in enumerate(refined_lines) if line.strip()]

        max_lines = max(len(orig_non_empty), len(ref_non_empty), 1)

        all_pairs = []
        for orig_idx, (orig_line_num, orig_content) in enumerate(orig_non_empty):
            for ref_idx, (ref_line_num, ref_content) in enumerate(ref_non_empty):
                content_sim = self._get_line_similarity(orig_content, ref_content)
                if content_sim >= 0.4:
                    context_sim = self._get_context_similarity(
                        orig_idx, ref_idx, orig_non_empty, ref_non_empty
                    )
                    position_diff = abs(orig_idx - ref_idx)
                    position_sim = 1.0 - (position_diff / max_lines)

                    total_score = content_sim * 0.6 + context_sim * 0.2 + position_sim * 0.2
                    all_pairs.append((orig_idx, ref_idx, orig_line_num, ref_line_num, total_score))

        all_pairs.sort(key=lambda x: x[4], reverse=True)

        matched_pairs = []
        used_orig = set()
        used_ref = set()

        def would_cross(orig_num: int, ref_num: int) -> bool:
            """Check if a new match would cross any existing match."""
            for prev_orig, prev_ref in matched_pairs:
                if (orig_num > prev_orig and ref_num < prev_ref) or \
                   (orig_num < prev_orig and ref_num > prev_ref):
                    return True
            return False

        for orig_idx, ref_idx, orig_line_num, ref_line_num, total_score in all_pairs:
            if orig_idx in used_orig or ref_idx in used_ref:
                continue

            if would_cross(orig_line_num, ref_line_num):
                continue

            matched_pairs.append((orig_line_num, ref_line_num))
            used_orig.add(orig_idx)
            used_ref.add(ref_idx)

        unmatched_original = [orig_non_empty[i][0] for i in range(len(orig_non_empty)) if i not in used_orig]
        unmatched_refined = [ref_non_empty[j][0] for j in range(len(ref_non_empty)) if j not in used_ref]

        matched_pairs.sort(key=lambda x: x[0])

        return matched_pairs, unmatched_original, unmatched_refined

    def merge_remove_lines_operations(self, operations: List[Dict]) -> List[Dict]:
        """Merge remove operations: use remove_lines(start, end) uniformly (start==end for single lines)."""
        remove_line_numbers = []
        other_operations = []
        removed_contents_dict = {}

        for op in operations:
            if op['operation'] == 'remove_lines':
                remove_line_numbers.extend(op['line_numbers'])
                for line_num, content in zip(op['line_numbers'], op.get('removed_content', [])):
                    removed_contents_dict[line_num] = content
            else:
                other_operations.append(op)

        result_operations = []
        if remove_line_numbers:
            unique_line_numbers = sorted(set(remove_line_numbers))
            i = 0
            while i < len(unique_line_numbers):
                start = end = unique_line_numbers[i]
                while i + 1 < len(unique_line_numbers) and unique_line_numbers[i + 1] == unique_line_numbers[i] + 1:
                    i += 1
                    end = unique_line_numbers[i]

                result_operations.append({
                    'operation': 'remove_lines',
                    'start_line': start,
                    'end_line': end,
                    'description': f"Remove line {start}{'-' + str(end) if end != start else ''}",
                    'removed_content': [removed_contents_dict.get(n, '') for n in range(start, end + 1)]
                })
                i += 1

        return result_operations + other_operations

    def analyze_char_operations(self, orig_line: str, ref_line: str, line_num: int) -> List[Dict]:
        """Analyze character-level changes within a line, producing replace_str operations."""
        matcher = difflib.SequenceMatcher(None, orig_line, ref_line)
        operations = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            elif tag == 'delete':
                removed_str = orig_line[i1:i2]
                context_before, target, context_after = self._make_context_unique_for_position(orig_line, i1, i2)
                operations.append({
                    'operation': 'replace_str',
                    'line_number': line_num,
                    'search_content': context_before + target + context_after,
                    'replace_content': context_before + context_after,
                    'description': f"Line {line_num}: delete string {repr(removed_str)}"
                })
            elif tag == 'insert':
                inserted_str = ref_line[j1:j2]
                context_before, context_after = self._make_insert_context_unique(orig_line, i1)

                operations.append({
                    'operation': 'replace_str',
                    'line_number': line_num,
                    'search_content': context_before + context_after,
                    'replace_content': context_before + inserted_str + context_after,
                    'description': f"Line {line_num}: insert string {repr(inserted_str)}"
                })
            elif tag == 'replace':
                old_str, new_str = orig_line[i1:i2], ref_line[j1:j2]
                context_before, target, context_after = self._make_context_unique_for_position(orig_line, i1, i2)
                operations.append({
                    'operation': 'replace_str',
                    'line_number': line_num,
                    'search_content': context_before + target + context_after,
                    'replace_content': context_before + new_str + context_after,
                    'description': f"Line {line_num}: replace {repr(old_str)} -> {repr(new_str)}"
                })

        return operations

    def _make_context_unique_for_position(self, line: str, start: int, end: int) -> Tuple[str, str, str]:
        """Generate context ensuring find() locates the correct position.

        Not only requires search_content to be unique, but also verifies
        that find() returns the correct position.
        """
        target = line[start:end]
        max_possible_len = max(start, len(line) - end)

        for context_len in range(0, max_possible_len + 1):
            before = line[max(0, start - context_len):start]
            after = line[end:min(len(line), end + context_len)]
            search_content = before + target + after

            found_pos = line.find(search_content)
            expected_pos = start - len(before)

            if found_pos == expected_pos:
                return before, target, after

        return line[0:start], target, line[end:]

    def _make_insert_context_unique(self, line: str, insert_pos: int) -> Tuple[str, str]:
        """Generate unique context for insertion, ensuring find() locates the correct insertion point."""
        max_context = max(insert_pos, len(line) - insert_pos, 1)

        for context_len in range(1, max_context + 1):
            before = line[max(0, insert_pos - context_len):insert_pos]
            after = line[insert_pos:min(len(line), insert_pos + context_len)]
            search_content = before + after

            found_pos = line.find(search_content)
            expected_pos = insert_pos - len(before)

            if found_pos == expected_pos:
                return before, after

        return line[:insert_pos], line[insert_pos:]

    def reorder_operations_for_execution(self, operations: List[Dict]) -> List[Dict]:
        """Reorder operations: sort by line number, within same line by search_content length (descending)."""
        str_ops = [op for op in operations if op['operation'] == 'replace_str']
        other_ops = [op for op in operations if op['operation'] != 'replace_str']
        str_ops.sort(key=lambda op: (op['line_number'], -len(op['search_content'])))
        return str_ops + other_ops

    def extract_modifications(self, search: str, replace: str, pos_in_line: int) -> List[Tuple[int, int, str]]:
        """Extract actual modifications from search_content and replace_content.

        Args:
            search: search_content
            replace: replace_content
            pos_in_line: starting position of search_content in the original line

        Returns:
            List of modifications: [(abs_start_in_line, abs_end_in_line, new_content), ...]
        """
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

    def merge_overlapping_ops(self, ops: List[Dict], orig_line: str, line_num: int, max_merged_len: int = 30) -> List[Dict]:
        """Merge multiple nearby operations into a single operation.

        Uses minimum necessary range as search_content.
        Does not merge if the resulting length exceeds max_merged_len.

        Args:
            ops: list of operations to merge
            orig_line: original line content
            line_num: line number
            max_merged_len: max length of merged search_content

        Returns:
            Merged operation list (may be a single merged op or the original list)
        """
        all_modifications = []

        for op in ops:
            search = op['search_content']
            replace = op['replace_content']

            pos = orig_line.find(search)
            if pos == -1:
                continue

            mods = self.extract_modifications(search, replace, pos)
            all_modifications.extend(mods)

        if not all_modifications:
            return ops

        all_modifications.sort(key=lambda x: x[0])

        merged_modifications = []
        for start, end, content in all_modifications:
            if merged_modifications:
                last_start, last_end, last_content = merged_modifications[-1]
                if start < last_end:
                    new_start = last_start
                    new_end = max(last_end, end)
                    overlap_in_last = start - last_start
                    new_content = last_content[:overlap_in_last] + content
                    merged_modifications[-1] = (new_start, new_end, new_content)
                    continue
            merged_modifications.append((start, end, content))

        if not merged_modifications:
            return ops

        min_start = merged_modifications[0][0]
        max_end = max(m[1] for m in merged_modifications)

        merged_len = max_end - min_start
        if merged_len > max_merged_len:
            return ops

        context_before, target, context_after, _ = self.make_context_unique_in_line(
            orig_line, min_start, max_end
        )
        search_content = context_before + target + context_after

        new_start = min_start - len(context_before)

        replace_content = search_content
        for start, end, new_content in reversed(merged_modifications):
            rel_start = start - new_start
            rel_end = end - new_start
            replace_content = replace_content[:rel_start] + new_content + replace_content[rel_end:]

        if replace_content == search_content:
            return ops

        return [{
            'operation': 'replace_str',
            'line_number': line_num,
            'search_content': search_content,
            'replace_content': replace_content,
            'description': f"Line {line_num}: merge {len(ops)} nearby replace operations"
        }]

    def get_ops_distance(self, op_a: Dict, op_b: Dict, orig_line: str) -> int:
        """Compute distance between two operations (based on actual modification positions)."""
        pos_a = orig_line.find(op_a['search_content'])
        if pos_a == -1:
            return float('inf')
        mods_a = self.extract_modifications(op_a['search_content'], op_a['replace_content'], pos_a)

        pos_b = orig_line.find(op_b['search_content'])
        if pos_b == -1:
            return float('inf')
        mods_b = self.extract_modifications(op_b['search_content'], op_b['replace_content'], pos_b)

        if not mods_a:
            mods_a = [(pos_a, pos_a + len(op_a['search_content']), '')]
        if not mods_b:
            mods_b = [(pos_b, pos_b + len(op_b['search_content']), '')]

        min_gap = float('inf')
        for start_a, end_a, _ in mods_a:
            for start_b, end_b, _ in mods_b:
                if end_a <= start_b:
                    gap = start_b - end_a
                elif end_b <= start_a:
                    gap = start_a - end_b
                else:
                    gap = 0
                min_gap = min(min_gap, gap)

        return min_gap

    def greedy_merge_ops(self, ops: List[Dict], orig_line: str, line_num: int,
                         max_gap: int = 30, max_merged_len: int = 20) -> List[Dict]:
        """Greedy merge: merge operations by distance priority.

        Args:
            ops: operation list
            orig_line: original line content
            line_num: line number
            max_gap: maximum allowed distance between operations
            max_merged_len: maximum search_content length after merging

        Returns:
            Merged operation list
        """
        if len(ops) <= 1:
            return ops

        n = len(ops)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.get_ops_distance(ops[i], ops[j], orig_line)
                if dist <= max_gap:
                    distances.append((dist, i, j))

        distances.sort(key=lambda x: x[0])

        # Union-Find for tracking merged operation groups
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        group_ranges = {}

        for op_idx, op in enumerate(ops):
            pos = orig_line.find(op['search_content'])
            if pos != -1:
                mods = self.extract_modifications(op['search_content'], op['replace_content'], pos)
                if mods:
                    min_start = min(m[0] for m in mods)
                    max_end = max(m[1] for m in mods)
                else:
                    min_start = pos
                    max_end = pos + len(op['search_content'])
                group_ranges[op_idx] = (min_start, max_end)

        for dist, i, j in distances:
            root_i, root_j = find(i), find(j)
            if root_i == root_j:
                continue

            if root_i not in group_ranges or root_j not in group_ranges:
                continue

            range_i = group_ranges[root_i]
            range_j = group_ranges[root_j]
            new_min = min(range_i[0], range_j[0])
            new_max = max(range_i[1], range_j[1])
            new_len = new_max - new_min

            if new_len <= max_merged_len:
                union(i, j)
                new_root = find(i)
                group_ranges[new_root] = (new_min, new_max)
                if root_i != new_root and root_i in group_ranges:
                    del group_ranges[root_i]
                if root_j != new_root and root_j in group_ranges:
                    del group_ranges[root_j]

        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(ops[i])

        result = []
        for root, group_ops in groups.items():
            if len(group_ops) == 1:
                result.append(group_ops[0])
            else:
                merged = self.merge_overlapping_ops(group_ops, orig_line, line_num)
                result.extend(merged)

        return result

    def merge_same_line_replace_operations(self, operations: List[Dict], original_lines: List[str]) -> List[Dict]:
        """Merge replace_str operations on the same line.

        Uses greedy distance-based merging to reduce operation count
        while keeping each operation's content manageable.

        Args:
            operations: operation list
            original_lines: original text split into lines (with newlines)

        Returns:
            Processed operation list
        """
        replace_ops = [op for op in operations if op['operation'] == 'replace_str']
        other_ops = [op for op in operations if op['operation'] != 'replace_str']

        if not replace_ops:
            return operations

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

            merged_result = self.greedy_merge_ops(ops, orig_line, line_num)
            merged_ops.extend(merged_result)

        return merged_ops + other_ops

    def analyze_line_operations(self, original: str, refined: str) -> List[Dict]:
        """Analyze original and refined text, producing a list of line-based operations."""
        if original == refined:
            return []

        if refined.strip() == '[Content valueless, deleted]':
            return [{
                'operation': 'remove_all',
                'description': 'Content is valueless, remove entirely'
            }]

        original_lines = original.splitlines(keepends=True)
        refined_lines = refined.splitlines(keepends=True)

        matched_pairs, unmatched_original, unmatched_refined = self.match_lines(original_lines, refined_lines)

        operations = []

        # 1. Matched line pairs -> generate replace_str operations
        for orig_line_num, ref_line_num in matched_pairs:
            orig_line = original_lines[orig_line_num - 1]
            ref_line = refined_lines[ref_line_num - 1]
            if orig_line != ref_line:
                char_ops = self.analyze_char_operations(orig_line, ref_line, orig_line_num)
                operations.extend(char_ops)

        # 2. Unmatched original lines -> generate remove_lines operations
        if unmatched_original:
            operations.append({
                'operation': 'remove_lines',
                'line_numbers': unmatched_original,
                'description': f"Remove line {', '.join(map(str, unmatched_original[:3]))}{'...' if len(unmatched_original) > 3 else ''}",
                'removed_content': [original_lines[i-1] for i in unmatched_original]
            })

        # 3. Unmatched refined lines -> generate add_line operations
        if unmatched_refined:
            refined_to_orig = {ref: orig for orig, ref in matched_pairs}

            insert_after_counter = {}
            insert_before_counter = {}

            for ref_line_num in sorted(unmatched_refined):
                content = refined_lines[ref_line_num - 1]

                prev_orig = 0
                for ref_num in range(ref_line_num - 1, 0, -1):
                    if ref_num in refined_to_orig:
                        prev_orig = refined_to_orig[ref_num]
                        break

                next_orig = 0
                for ref_num in range(ref_line_num + 1, len(refined_lines) + 1):
                    if ref_num in refined_to_orig:
                        next_orig = refined_to_orig[ref_num]
                        break

                if prev_orig > 0:
                    base_position = prev_orig
                    sub_index = insert_after_counter.get(base_position, 0) + 1
                    insert_after_counter[base_position] = sub_index
                    desc = f"Insert new line #{sub_index} after line {base_position}"
                elif next_orig > 0:
                    base_position = next_orig - 1
                    sub_index = insert_before_counter.get(next_orig, 0) + 1
                    insert_before_counter[next_orig] = sub_index
                    desc = f"Insert new line #{sub_index} before line {next_orig}"
                else:
                    base_position = 0
                    sub_index = insert_before_counter.get(0, 0) + 1
                    insert_before_counter[0] = sub_index
                    desc = f"Insert new line #{sub_index} at beginning of file"

                operations.append({
                    'operation': 'add_line',
                    'base_position': base_position,
                    'sub_index': sub_index,
                    'content': content,
                    'description': desc
                })

        operations = self.merge_remove_lines_operations(operations)
        operations = self.merge_same_line_replace_operations(operations, original_lines)
        operations = self.filter_meaningful_operations(operations)
        operations = self.reorder_operations_for_execution(operations)

        return operations

    def filter_data(self, operations: List[Dict]) -> bool:
        """Check whether data meets filtering criteria.

        Returns:
            True: keep this data
            False: filter out this data
        """
        if len(operations) >= 20:
            return False

        add_line_count = 0
        for op in operations:
            if op['operation'] == 'replace_str':
                if len(op['search_content']) >= 150:
                    return False
                if len(op['replace_content']) >= 150:
                    return False

            if op['operation'] == 'add_line':
                add_line_count += 1
                if len(op['content']) >= 200:
                    return False

        if add_line_count >= 10:
            return False

        return True

    def add_line_id_markers(self, content: str) -> str:
        """Add line number markers to each line of the original content."""
        lines = content.splitlines()
        marked_lines = []
        for i, line in enumerate(lines, start=1):
            marked_lines.append(f"<lid:{i}>{line}")
        return '\n'.join(marked_lines)

    def generate_simple_function(self, operations: List[Dict]) -> str:
        """Generate simplified function code from operations."""
        if not operations:
            return "keep_all()"

        if len(operations) == 1 and operations[0]['operation'] == 'remove_all':
            return "remove_all()"

        func_lines = []
        for op in operations:
            if op['operation'] == 'remove_lines':
                func_lines.append(f"remove_lines({op['start_line']}, {op['end_line']})")
            elif op['operation'] == 'add_line':
                content = op['content'] if op['content'].endswith('\n') else op['content'] + '\n'
                func_lines.append(f"add_line({op['base_position']}, {op['sub_index']}, {repr(content)})")
            elif op['operation'] == 'replace_str':
                func_lines.append(f"replace_str({op['line_number']}, {repr(op['search_content'])}, {repr(op['replace_content'])})")

        return '\n'.join(func_lines)


def get_combo_name(operations: List[Dict]) -> str:
    if not operations:
        return "keep_all"
    if len(operations) == 1 and operations[0]['operation'] == 'remove_all':
        return "remove_all"
    return '_'.join(sorted(set(op['operation'] for op in operations)))


def process_single_file(parquet_file: Path, input_dir: Path, output_dir: Path) -> Dict:
    """Process a single parquet file (for multiprocessing)."""
    stats = {"processed": 0, "filtered": 0, "skipped": 0, "failed": 0}
    relative_path = parquet_file.relative_to(input_dir)
    output_json = output_dir / relative_path.with_suffix('.jsonl')
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if output_json.exists():
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                stats["processed"] = sum(1 for _ in f)
        except Exception:
            pass
        return stats

    cleaner = BatchDataCleaner()
    results = []

    try:
        pf = pq.ParquetFile(parquet_file)

        for batch in pf.iter_batches(batch_size=50):
            df = batch.to_pandas()
            required = ['original_content', 'refined_content', 'refinement_success']
            if any(col not in df.columns for col in required):
                continue

            for _, row in df.iterrows():
                if str(row['refinement_success']).strip().lower() not in ('yes', 'true', '1'):
                    stats["skipped"] += 1
                    continue

                raw_original = row['original_content']
                raw_refined = row['refined_content']
                if raw_original is None or raw_refined is None:
                    stats["skipped"] += 1
                    continue
                original_content = str(raw_original)
                refined_content = str(raw_refined)
                if original_content in ('nan', 'None', '') or refined_content in ('nan', 'None', ''):
                    stats["skipped"] += 1
                    continue

                try:
                    operations = cleaner.analyze_line_operations(original_content, refined_content)

                    if not cleaner.filter_data(operations):
                        stats["filtered"] += 1
                        continue

                    combo_name = get_combo_name(operations)
                    marked_content = cleaner.add_line_id_markers(original_content)
                    simple_func = cleaner.generate_simple_function(operations)

                    result = {
                        "combo": combo_name,
                        "messages": [
                            {"role": "user", "content": marked_content},
                            {"role": "assistant", "content": "<think>\n\n</think>\n\n" + simple_func}
                        ]
                    }
                    results.append(result)
                    stats["processed"] += 1

                except Exception as e:
                    stats["failed"] += 1
                    if stats["failed"] <= 3:
                        logger.warning(f"Error processing data ({relative_path}): {type(e).__name__}: {e}")

        if results:
            with open(output_json, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

        logger.info(f"Done: {relative_path}, processed={stats['processed']}, filtered={stats['filtered']}, skipped={stats['skipped']}, failed={stats['failed']}")

    except Exception as e:
        logger.error(f"Failed to process file {parquet_file.name}: {e}")

    return stats


def merge_by_combo(output_dir: Path):
    """Merge per-parquet JSONL files by function combo into by_combo/ directory."""
    combo_dir = output_dir / "by_combo"
    combo_dir.mkdir(parents=True, exist_ok=True)

    combo_file_handles = {}
    combo_counts = defaultdict(int)

    raw_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != 'by_combo']
    all_jsonl_files = []
    for d in raw_dirs:
        all_jsonl_files.extend(sorted(d.rglob("*.jsonl")))

    logger.info(f"Merging by function combo, {len(all_jsonl_files)} files total...")

    for jsonl_file in all_jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    combo = data.pop("combo", "unknown")
                    clean_line = json.dumps(data, ensure_ascii=False)

                    if combo not in combo_file_handles:
                        combo_file_handles[combo] = open(
                            combo_dir / f"{combo}.jsonl", 'w', encoding='utf-8')
                    combo_file_handles[combo].write(clean_line + '\n')
                    combo_counts[combo] += 1
                except Exception:
                    pass

    for fh in combo_file_handles.values():
        fh.close()

    logger.info("=" * 50)
    logger.info("Merge by function combo complete:")
    total = sum(combo_counts.values())
    for combo in sorted(combo_counts.keys()):
        cnt = combo_counts[combo]
        pct = cnt / total * 100 if total > 0 else 0
        logger.info(f"  {combo}.jsonl: {cnt} entries ({pct:.2f}%)")
    logger.info(f"  Total: {total} entries")
    logger.info(f"  Output directory: {combo_dir}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="LAM + DCR Function Construction: convert (original, refined) text pairs "
                    "into structured cleaning-function training data."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input parquet files with columns: "
                             "original_content, refined_content, refinement_success")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write output JSONL training data")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of parallel worker processes (default: 64)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files found")
        return

    total_files = len(parquet_files)
    logger.info(f"Found {total_files} parquet files")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    start_time = datetime.now()

    total_stats = {"processed": 0, "filtered": 0, "skipped": 0, "failed": 0}
    process_func = partial(process_single_file, input_dir=input_dir, output_dir=output_dir)

    with Pool(processes=args.num_workers) as pool:
        for i, stats in enumerate(pool.imap_unordered(process_func, parquet_files), 1):
            for key in total_stats:
                total_stats[key] += stats[key]
            if i % 500 == 0 or i == total_files:
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = i / elapsed if elapsed > 0 else 0
                eta = (total_files - i) / speed if speed > 0 else 0
                logger.info(f"Progress: {i}/{total_files} ({i/total_files*100:.1f}%) | "
                           f"Speed: {speed:.1f} files/sec | ETA: {eta/60:.1f} min | "
                           f"Processed: {total_stats['processed']} Filtered: {total_stats['filtered']}")

    end_time = datetime.now()

    logger.info("=" * 50)
    logger.info("Processing complete")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Processed: {total_stats['processed']} entries")
    logger.info(f"Filtered: {total_stats['filtered']} entries")
    logger.info(f"Skipped (not refined): {total_stats['skipped']} entries")
    logger.info(f"Failed: {total_stats['failed']} entries")
    logger.info(f"Elapsed time: {end_time - start_time}")
    logger.info("=" * 50)

    logger.info("Starting merge by combo...")
    merge_by_combo(output_dir)

    logger.info("All done!")


if __name__ == "__main__":
    main()
