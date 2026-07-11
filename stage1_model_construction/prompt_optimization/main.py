#!/usr/bin/env python3
"""
End-to-End Data Refinement Prompt Auto-Optimizer
=================================================

For each dataset in seed_data_sample_split_12k:
  1. Independently sample 100 texts for optimisation + 100 for testing
  2. Profile the dataset's dirty-data patterns
  3. Iteratively optimise a refinement prompt via multi-round LLM feedback
  4. Evaluate the best prompt on the independent test set
  5. Persist everything (samples, intermediate outputs, prompts, scores)

Usage:
  # Optimise all datasets
  python main.py --api-key <KEY>

  # Optimise specific datasets
  python main.py --api-key <KEY> --datasets fineweb stack_overflow

  # Customise parameters
  python main.py --api-key <KEY> --max-iterations 3 --batch-size 8

  # Resume (skip datasets whose outputs already exist)
  python main.py --api-key <KEY> --resume
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from api_client import LLMClient
from data_sampler import list_datasets, sample_dataset_pair
from optimizer import PromptOptimizer


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_reference_prompts() -> dict[str, str]:
    prompts: dict[str, str] = {}
    for fname in sorted(os.listdir(cfg.PROMPT_REFERENCE_DIR)):
        if fname.endswith(".txt"):
            with open(
                os.path.join(cfg.PROMPT_REFERENCE_DIR, fname), "r", encoding="utf-8"
            ) as f:
                prompts[fname] = f.read()
    return prompts


def load_base_prompt() -> str:
    with open(cfg.BASE_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-End Data Refinement Prompt Auto-Optimizer"
    )
    p.add_argument("--api-key", type=str, default=None,
                    help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    p.add_argument("--datasets", nargs="*", default=None,
                    help="Specific dataset names to optimise (default: all)")
    p.add_argument("--max-iterations", type=int, default=None,
                    help=f"Max optimisation iterations (default: {cfg.MAX_ITERATIONS})")
    p.add_argument("--batch-size", type=int, default=None,
                    help=f"Refinement batch size per iteration (default: {cfg.REFINE_BATCH_SIZE})")
    p.add_argument("--opt-samples", type=int, default=None,
                    help=f"Optimisation samples per dataset (default: {cfg.OPT_SAMPLE_SIZE})")
    p.add_argument("--test-samples", type=int, default=None,
                    help=f"Test samples per dataset (default: {cfg.TEST_SAMPLE_SIZE})")
    p.add_argument("--resume", action="store_true",
                    help="Skip datasets that already have optimized_prompt.txt")
    return p.parse_args()


def apply_overrides(args):
    if args.api_key:
        cfg.API_KEY = args.api_key
    if args.opt_samples is not None:
        cfg.OPT_SAMPLE_SIZE = args.opt_samples
    if args.test_samples is not None:
        cfg.TEST_SAMPLE_SIZE = args.test_samples

    if args.batch_size is not None:
        cfg.REFINE_BATCH_SIZE = args.batch_size
    if args.max_iterations is not None:
        cfg.MAX_ITERATIONS = args.max_iterations
    else:
        # Ensure MAX_ITERATIONS × REFINE_BATCH_SIZE == OPT_SAMPLE_SIZE
        cfg.MAX_ITERATIONS = max(1, cfg.OPT_SAMPLE_SIZE // cfg.REFINE_BATCH_SIZE)


def main():
    args = parse_args()
    apply_overrides(args)

    setup_logging()
    logger = logging.getLogger("main")

    logger.info("=" * 70)
    logger.info("  End-to-End Data Refinement Prompt Auto-Optimizer")
    logger.info("=" * 70)
    logger.info("  Refine model     : %s (temp=%.1f)", cfg.REFINE_MODEL, cfg.REFINE_TEMPERATURE)
    logger.info("  Judge model      : %s (temp=%.1f)", cfg.JUDGE_MODEL, cfg.JUDGE_TEMPERATURE)
    logger.info("  Opt samples      : %d", cfg.OPT_SAMPLE_SIZE)
    logger.info("  Test samples     : %d", cfg.TEST_SAMPLE_SIZE)
    logger.info("  Max iterations   : %d", cfg.MAX_ITERATIONS)
    logger.info("  Batch size       : %d", cfg.REFINE_BATCH_SIZE)
    logger.info("  Coverage         : %d × %d = %d / %d opt samples",
                cfg.MAX_ITERATIONS, cfg.REFINE_BATCH_SIZE,
                cfg.MAX_ITERATIONS * cfg.REFINE_BATCH_SIZE, cfg.OPT_SAMPLE_SIZE)
    logger.info("  Max tokens (out) : %d", cfg.API_MAX_TOKENS)
    logger.info("=" * 70)

    if not cfg.API_KEY:
        logger.error(
            "No API key provided. Set DEEPSEEK_API_KEY env var or pass --api-key"
        )
        sys.exit(1)

    reference_prompts = load_reference_prompts()
    base_prompt = load_base_prompt()
    logger.info(
        "Loaded %d reference prompts, base prompt len=%d",
        len(reference_prompts),
        len(base_prompt),
    )

    client = LLMClient()
    optimizer = PromptOptimizer(client, reference_prompts)

    all_ds = list_datasets()
    targets = args.datasets if args.datasets else all_ds
    invalid = set(targets) - set(all_ds)
    if invalid:
        logger.error("Unknown datasets: %s", invalid)
        sys.exit(1)

    if args.resume:
        already = [
            d
            for d in targets
            if os.path.isfile(
                os.path.join(cfg.OUTPUT_DIR, d, "optimized_prompt.txt")
            )
        ]
        if already:
            logger.info(
                "Resuming — skipping %d completed: %s", len(already), already
            )
            targets = [d for d in targets if d not in already]

    logger.info("Datasets to optimise (%d): %s", len(targets), targets)

    results = []
    for idx, ds in enumerate(targets):
        logger.info("\n" + "#" * 70)
        logger.info("# [%d/%d] %s", idx + 1, len(targets), ds)
        logger.info("#" * 70)

        try:
            opt_set, test_set = sample_dataset_pair(
                ds,
                opt_n=cfg.OPT_SAMPLE_SIZE,
                test_n=cfg.TEST_SAMPLE_SIZE,
            )
            logger.info(
                "  Sampled: opt=%d  test=%d (independent, no overlap)",
                len(opt_set),
                len(test_set),
            )

            result = optimizer.optimize_dataset(
                dataset_name=ds,
                optimization_set=opt_set,
                test_set=test_set,
                base_prompt=base_prompt,
            )
            results.append(result)

        except Exception as exc:
            logger.error("Dataset %s FAILED: %s", ds, exc, exc_info=True)
            results.append({"dataset": ds, "error": str(exc)})

    # ---- overall summary ----
    summary = {
        "run_time": datetime.now().isoformat(),
        "config": {
            "refine_model": cfg.REFINE_MODEL,
            "judge_model": cfg.JUDGE_MODEL,
            "refine_temperature": cfg.REFINE_TEMPERATURE,
            "judge_temperature": cfg.JUDGE_TEMPERATURE,
            "max_iterations": cfg.MAX_ITERATIONS,
            "batch_size": cfg.REFINE_BATCH_SIZE,
            "opt_samples": cfg.OPT_SAMPLE_SIZE,
            "test_samples": cfg.TEST_SAMPLE_SIZE,
        },
        "api_stats": client.stats,
        "results": results,
    }
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(cfg.OUTPUT_DIR, "optimization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("  OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    for r in results:
        if "error" in r:
            logger.info("  %-50s  FAILED  %s", r["dataset"], r["error"][:60])
        else:
            logger.info(
                "  %-50s  score=%.2f  iters=%d",
                r["dataset"],
                r["final_score"],
                r["iterations"],
            )
    logger.info("  Summary → %s", summary_path)
    logger.info(
        "  API calls: %d  tokens(est): %d",
        client.stats["total_calls"],
        client.stats["total_tokens_est"],
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
