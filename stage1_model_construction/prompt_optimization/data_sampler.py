"""
Randomly sample records from each parquet-based dataset.

Key design: optimisation set and test set are drawn with **different
random seeds** so they are fully independent (no overlap).
"""

import os
import random
import logging

import pyarrow.parquet as pq

import config as cfg

logger = logging.getLogger(__name__)


def list_datasets() -> list[str]:
    """Return sorted list of dataset directory names."""
    return sorted(
        d
        for d in os.listdir(cfg.SEED_DATA_DIR)
        if os.path.isdir(os.path.join(cfg.SEED_DATA_DIR, d))
    )


def _parquet_files(dataset_name: str) -> list[str]:
    folder = os.path.join(cfg.SEED_DATA_DIR, dataset_name)
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".parquet")
    )


def _count_rows(files: list[str]) -> tuple[list[int], int]:
    counts = []
    total = 0
    for pf in files:
        c = pq.read_metadata(pf).num_rows
        counts.append(c)
        total += c
    return counts, total


def _fetch_by_global_indices(
    files: list[str],
    file_row_counts: list[int],
    indices: list[int],
) -> list[str]:
    """Load only the parquet files that contain requested rows."""
    indices = sorted(indices)
    texts: list[str] = []
    cum = 0
    ptr = 0
    for pf, cnt in zip(files, file_row_counts):
        local = []
        while ptr < len(indices) and indices[ptr] < cum + cnt:
            local.append(indices[ptr] - cum)
            ptr += 1
        if local:
            col = pq.read_table(pf, columns=[cfg.TEXT_COLUMN]).column(cfg.TEXT_COLUMN)
            for idx in local:
                texts.append(col[idx].as_py())
        cum += cnt
        if ptr >= len(indices):
            break
    return texts


def sample_dataset_pair(
    dataset_name: str,
    opt_n: int = cfg.OPT_SAMPLE_SIZE,
    test_n: int = cfg.TEST_SAMPLE_SIZE,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Draw two **non-overlapping** samples from *dataset_name* using a
    single random draw then split.

    Returns (optimisation_set, test_set).
    """
    files = _parquet_files(dataset_name)
    if not files:
        raise FileNotFoundError(f"No parquet files in dataset '{dataset_name}'")

    file_row_counts, total_rows = _count_rows(files)

    need = opt_n + test_n
    if need > total_rows:
        raise ValueError(
            f"Dataset '{dataset_name}' has only {total_rows} rows, "
            f"but {opt_n}+{test_n}={need} requested"
        )

    rng = random.Random(seed)
    all_indices = rng.sample(range(total_rows), need)
    opt_indices = all_indices[:opt_n]
    test_indices = all_indices[opt_n:]

    logger.info(
        "Dataset %-45s  rows=%d  files=%d  opt=%d  test=%d",
        dataset_name, total_rows, len(files), opt_n, test_n,
    )

    opt_texts = _fetch_by_global_indices(files, file_row_counts, opt_indices)
    test_texts = _fetch_by_global_indices(files, file_row_counts, test_indices)

    rng.shuffle(opt_texts)
    rng.shuffle(test_texts)

    return opt_texts, test_texts
