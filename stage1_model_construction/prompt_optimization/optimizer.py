"""
Core prompt-optimization agent.

For every dataset the loop is:
  0. Profile the dataset to understand its dirty-data patterns
  1. Refine a batch with the current prompt
  2. Judge the refinements (score + issues)
  3. Feed failures + dataset profile into a meta-optimizer that rewrites the prompt
  4. Validate the new prompt on a held-out mini-batch
  5. Repeat until convergence or MAX_ITERATIONS

After the loop the best prompt is evaluated on the **independent** 100-sample
test set and everything (samples, intermediate refinements, prompts) is
persisted to OUTPUT_DIR/<dataset>/.
"""

import os
import re
import json
import time
import random
import logging
from datetime import datetime

from api_client import LLMClient
from evaluator import RefinementEvaluator
import config as cfg

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Dataset profiling prompt                                             #
# ------------------------------------------------------------------ #
PROFILE_SYSTEM_PROMPT = """\
You are a data-quality analyst. You will receive 15 sample texts from a web
dataset called "{dataset_name}".

Produce a JSON profile with these fields (no markdown fences):
{{
  "dominant_language": "en|zh|mixed|...",
  "content_types": ["article", "forum_post", "code_qa", "product_page", ...],
  "common_noise_patterns": ["navigation bars", "cookie banners", ...],
  "has_code_blocks": true/false,
  "has_tables": true/false,
  "has_math": true/false,
  "avg_quality_impression": "high|medium|low",
  "special_notes": "free text with anything else noteworthy"
}}
Be concise. Output ONLY the JSON object.
"""


# ------------------------------------------------------------------ #
# Meta-optimizer system prompt                                         #
# ------------------------------------------------------------------ #
META_SYSTEM_TEMPLATE = """\
You are a world-class Prompt Engineering Expert.

## Task Context — CRITICAL
This is a **pre-training corpus CLEANING pipeline**, NOT a rewriting pipeline.
The fundamental principles are:
- **Delete only**: remove noise (ads, navigation, HTML, boilerplate, gibberish,
  engagement bait, contact info, broken formatting).
- **Never rewrite**: clean text must be output character-for-character unchanged.
  No paraphrasing, no summarizing, no "style improvement", no rearranging.
- **Valueless → delete marker**: purely valueless content (all-ads, pure spam,
  gibberish) should produce exactly `[Content valueless, deleted]`.
- **Already-clean → pass through**: if the original needs zero changes, output
  it exactly as-is.

## Goal
Improve the **Data Cleaning Prompt** so it produces higher-quality outputs on
the "{dataset_name}" dataset.

## Dataset Profile
{dataset_profile}

## Optimization round {iteration}/{max_iter}

────────────────────────────────────────
## Current Prompt Being Optimized
<CURRENT_PROMPT>
{current_prompt}
</CURRENT_PROMPT>
────────────────────────────────────────

## Performance on this round's batch
- Average quality score : {avg_score} / 10
- Samples with MAJOR issues : {major_count} / {total_samples}
- Most frequent issues : {top_issues}

## Problematic Examples (original → model output → issues)
{examples_block}

## Reference Techniques from Other Prompts
{reference_block}

────────────────────────────────────────
## Instructions

1. **Diagnose** — identify root causes.  Pay special attention to:
   - Over-editing: did the model rewrite or rephrase clean text?
   - Under-cleaning: did noise survive that should have been removed?
   - Wrong deletions: was valuable content incorrectly removed?
2. **Prescribe** — tailor rules to the **Dataset Profile** (e.g. this
   dataset's specific noise patterns, content types, formatting conventions).
3. **Output** — the COMPLETE improved prompt between
   `<IMPROVED_PROMPT>` and `</IMPROVED_PROMPT>` tags.

Rules for the improved prompt:
- The output contract is: either the cleaned text or `[Content valueless, deleted]`.
- Emphasize the NO-REWRITE principle — the single most common failure mode.
- Preserve rules that already work; only add/modify for observed failures.
- Add dataset-specific guidance (e.g. how to handle code blocks, tables, Q&A
  threads, academic metadata, forum signatures, etc. for THIS dataset).
- If adding examples, keep them short and representative.
- Prompt language MUST remain English.
- Do NOT add conflicting rules.
"""


def _write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ------------------------------------------------------------------ #
# Optimizer class                                                      #
# ------------------------------------------------------------------ #
class PromptOptimizer:
    def __init__(
        self,
        client: LLMClient,
        reference_prompts: dict[str, str],
    ):
        self.client = client
        self.evaluator = RefinementEvaluator(client)
        self.reference_prompts = reference_prompts

    # ================================================================ #
    # Public entry point                                                #
    # ================================================================ #
    def optimize_dataset(
        self,
        dataset_name: str,
        optimization_set: list[str],
        test_set: list[str],
        base_prompt: str,
    ) -> dict:
        out_dir = os.path.join(cfg.OUTPUT_DIR, dataset_name)
        samples_dir = os.path.join(out_dir, "sampled_data")
        iters_dir = os.path.join(out_dir, "iterations")
        prompts_dir = os.path.join(out_dir, "prompts")
        for d in (out_dir, samples_dir, iters_dir, prompts_dir):
            os.makedirs(d, exist_ok=True)

        # ---- persist raw sampled data ----
        self._save_raw_samples(samples_dir, optimization_set, test_set)

        # ---- Step 0: profile dataset ----
        logger.info("=" * 64)
        logger.info("  Dataset : %s", dataset_name)
        logger.info("  Opt set : %d    Test set : %d", len(optimization_set), len(test_set))
        logger.info("=" * 64)

        dataset_profile = self._profile_dataset(dataset_name, optimization_set)
        _write_json(os.path.join(out_dir, "dataset_profile.json"), dataset_profile)
        logger.info("  [profile] %s", json.dumps(dataset_profile, ensure_ascii=False)[:300])

        # ---- save initial prompt (iteration 0) ----
        _write_text(os.path.join(prompts_dir, "prompt_iter_0_base.txt"), base_prompt)

        current_prompt = base_prompt

        log_record = {
            "dataset": dataset_name,
            "started_at": datetime.now().isoformat(),
            "base_prompt_len": len(base_prompt),
            "optimization_set_size": len(optimization_set),
            "test_set_size": len(test_set),
            "dataset_profile": dataset_profile,
            "iterations": [],
        }

        for it in range(cfg.MAX_ITERATIONS):
            t0 = time.time()
            logger.info("── iteration %d/%d ──", it + 1, cfg.MAX_ITERATIONS)

            iter_out = os.path.join(iters_dir, f"iter_{it + 1}")
            os.makedirs(iter_out, exist_ok=True)

            # --- 1. pick a batch (rotate through the optimization set) ---
            start = (it * cfg.REFINE_BATCH_SIZE) % len(optimization_set)
            batch = optimization_set[start : start + cfg.REFINE_BATCH_SIZE]
            if len(batch) < cfg.REFINE_BATCH_SIZE:
                batch += optimization_set[: cfg.REFINE_BATCH_SIZE - len(batch)]

            # --- 2. refine ---
            logger.info("  [refine]  %d samples (attempt 0) ...", len(batch))
            refinements = self._refine_batch(batch, current_prompt)
            batch, refinements = self._exclude_failures(batch, refinements)

            if not batch:
                logger.warning("  [skip]    all samples API-failed, skipping iteration")
                _write_text(os.path.join(prompts_dir, f"prompt_iter_{it + 1}.txt"), current_prompt)
                log_record["iterations"].append({
                    "iteration": it + 1, "batch_avg_score": 0, "inner_retries": 0,
                    "regression_score": None, "regression_retries": 0,
                    "top_issues": [["all_api_failed", cfg.REFINE_BATCH_SIZE]],
                    "prompt_len": len(current_prompt), "elapsed_sec": round(time.time() - t0, 1),
                })
                continue

            # --- persist attempt 0 ---
            self._save_iteration_pairs(iter_out, "refine_attempt_0", batch, refinements)

            # --- 3. evaluate ---
            logger.info("  [eval]    judging quality (attempt 0, %d samples) ...", len(batch))
            evals = self._evaluate_in_chunks(batch, refinements)
            summary = self.evaluator.summarize(evals)
            _write_json(os.path.join(iter_out, "eval_scores_attempt_0.json"), {
                "summary": summary,
                "per_sample": evals,
            })
            logger.info(
                "  [eval]    avg_score=%.2f  major=%d",
                summary["avg_score"],
                summary["major_count"],
            )

            # --- 4. inner loop: fix problems on THIS batch ---
            actual_retries = 0
            for retry in range(cfg.MAX_INNER_RETRIES):
                problems = [
                    (o, r, e)
                    for o, r, e in zip(batch, refinements, evals)
                    if e.get("total_score", 10) < 8
                    or e.get("severity") in ("major", "minor")
                ]

                all_fallback = all(
                    "evaluation_parse_failed" in e.get("issues", []) for e in evals
                )

                if not problems:
                    logger.info("  [fix]     all %d samples clean, moving on", len(batch))
                    break

                if all_fallback:
                    logger.warning("  [fix]     eval API failed, cannot fix this round")
                    break

                logger.info(
                    "  [fix]     retry %d/%d — %d problems remain (avg=%.2f) ...",
                    retry + 1, cfg.MAX_INNER_RETRIES, len(problems), summary["avg_score"],
                )

                new_prompt = self._meta_optimize(
                    dataset_name, dataset_profile, current_prompt,
                    problems, summary, it,
                )
                if new_prompt and len(new_prompt) > 200 and new_prompt != current_prompt:
                    current_prompt = new_prompt
                    logger.info("  [fix]     prompt updated (len=%d)", len(current_prompt))
                else:
                    logger.info("  [fix]     prompt unchanged, stopping inner loop")
                    break

                logger.info("  [fix]     re-refining %d samples ...", len(batch))
                refinements = self._refine_batch(batch, current_prompt)
                batch, refinements = self._exclude_failures(batch, refinements)
                if not batch:
                    logger.warning("  [fix]     all failed on re-refine, stopping inner loop")
                    break

                logger.info("  [fix]     re-evaluating ...")
                evals = self._evaluate_in_chunks(batch, refinements)
                summary = self.evaluator.summarize(evals)

                actual_retries += 1
                attempt_tag = f"attempt_{actual_retries}"
                self._save_iteration_pairs(iter_out, f"refine_{attempt_tag}", batch, refinements)
                _write_json(os.path.join(iter_out, f"eval_scores_{attempt_tag}.json"), {
                    "summary": summary,
                    "per_sample": evals,
                })
                logger.info(
                    "  [fix]     after retry %d: avg_score=%.2f  major=%d",
                    actual_retries, summary["avg_score"], summary["major_count"],
                )
            else:
                remaining = [
                    e for e in evals
                    if e.get("total_score", 10) < 8 or e.get("severity") in ("major", "minor")
                ]
                if remaining:
                    logger.warning(
                        "  [fix]     max retries reached, %d problems unresolved", len(remaining)
                    )

            # --- 5. regression check: test on 5 random historical samples ---
            regression_score = None
            regression_retries = 0
            if actual_retries > 0 and it > 0:
                past_end = it * cfg.REFINE_BATCH_SIZE
                past_pool = optimization_set[:past_end]
                reg_size = min(5, len(past_pool))
                reg_batch = random.sample(past_pool, reg_size)

                logger.info("  [regr]    testing %d historical samples ...", reg_size)
                reg_refs = self._refine_batch(reg_batch, current_prompt)
                reg_batch, reg_refs = self._exclude_failures(reg_batch, reg_refs)

                if not reg_batch:
                    logger.warning("  [regr]    all regression samples API-failed, skipping regression")
                else:
                    reg_evals = self._evaluate_in_chunks(reg_batch, reg_refs)
                    reg_summary = self.evaluator.summarize(reg_evals)
                    regression_score = reg_summary["avg_score"]

                    self._save_iteration_pairs(iter_out, "regression_attempt_0", reg_batch, reg_refs)
                    _write_json(os.path.join(iter_out, "regression_scores_attempt_0.json"), {
                        "summary": reg_summary,
                        "per_sample": reg_evals,
                    })

                    reg_problems = [
                        (o, r, e) for o, r, e in zip(reg_batch, reg_refs, reg_evals)
                        if e.get("total_score", 10) < 8 or e.get("severity") in ("major", "minor")
                    ]

                    if not reg_problems:
                        logger.info("  [regr]    no regression (avg=%.2f)", regression_score)
                    else:
                        logger.warning(
                            "  [regr]    regression detected! %d/%d degraded (avg=%.2f)",
                            len(reg_problems), reg_size, regression_score,
                        )
                        for reg_retry in range(cfg.MAX_INNER_RETRIES):
                            logger.info(
                                "  [regr]    fix retry %d/%d — feeding %d regressed samples to meta ...",
                                reg_retry + 1, cfg.MAX_INNER_RETRIES, len(reg_problems),
                            )
                            new_prompt = self._meta_optimize(
                                dataset_name, dataset_profile, current_prompt,
                                reg_problems, reg_summary, it,
                            )
                            if new_prompt and len(new_prompt) > 200 and new_prompt != current_prompt:
                                current_prompt = new_prompt
                                logger.info("  [regr]    prompt updated (len=%d)", len(current_prompt))
                            else:
                                logger.info("  [regr]    prompt unchanged, stopping")
                                break

                            reg_refs = self._refine_batch(reg_batch, current_prompt)
                            reg_batch, reg_refs = self._exclude_failures(reg_batch, reg_refs)
                            if not reg_batch:
                                logger.warning("  [regr]    all failed on re-refine, stopping")
                                break
                            reg_evals = self._evaluate_in_chunks(reg_batch, reg_refs)
                            reg_summary = self.evaluator.summarize(reg_evals)
                            regression_score = reg_summary["avg_score"]
                            regression_retries += 1

                            rtag = f"attempt_{regression_retries}"
                            self._save_iteration_pairs(iter_out, f"regression_{rtag}", reg_batch, reg_refs)
                            _write_json(os.path.join(iter_out, f"regression_scores_{rtag}.json"), {
                                "summary": reg_summary,
                                "per_sample": reg_evals,
                            })

                            reg_problems = [
                                (o, r, e) for o, r, e in zip(reg_batch, reg_refs, reg_evals)
                                if e.get("total_score", 10) < 8 or e.get("severity") in ("major", "minor")
                            ]
                            if not reg_problems:
                                logger.info("  [regr]    regression fixed (avg=%.2f)", regression_score)
                                break
                            logger.info(
                                "  [regr]    after retry %d: %d still degraded (avg=%.2f)",
                                regression_retries, len(reg_problems), regression_score,
                            )
                        else:
                            if reg_problems:
                                logger.warning("  [regr]    max retries, %d regressions unresolved", len(reg_problems))

            # --- save this iteration's prompt ---
            _write_text(
                os.path.join(prompts_dir, f"prompt_iter_{it + 1}.txt"),
                current_prompt,
            )

            elapsed = time.time() - t0
            iter_log = {
                "iteration": it + 1,
                "batch_avg_score": summary["avg_score"],
                "inner_retries": actual_retries,
                "regression_score": regression_score,
                "regression_retries": regression_retries,
                "top_issues": summary["top_issues"][:5],
                "prompt_len": len(current_prompt),
                "elapsed_sec": round(elapsed, 1),
            }
            log_record["iterations"].append(iter_log)

        # ---- final test: use the latest current_prompt (accumulated improvements) ----
        logger.info("── final test (%d independent samples) ──", len(test_set))

        test_dir = os.path.join(out_dir, "final_test")
        os.makedirs(test_dir, exist_ok=True)

        test_refs = self._refine_batch(test_set, current_prompt, concurrency=cfg.API_CONCURRENCY_TEST)
        test_evals = self._evaluate_in_chunks(test_set, test_refs, chunk_size=5, concurrency=cfg.API_CONCURRENCY_TEST)
        test_summary = self.evaluator.summarize(test_evals)
        logger.info("  final test score: %.2f / 10", test_summary["avg_score"])

        # --- persist ALL test pairs (full text, not truncated) ---
        self._save_iteration_pairs(test_dir, "test", test_set, test_refs)
        _write_json(os.path.join(test_dir, "test_eval_scores.json"), {
            "summary": test_summary,
            "per_sample": test_evals,
        })

        # ---- persist final artefacts ----
        log_record["finished_at"] = datetime.now().isoformat()
        log_record["final_test_score"] = test_summary["avg_score"]
        log_record["final_prompt_is"] = "latest current_prompt after 200 iterations"
        log_record["final_prompt_len"] = len(current_prompt)
        log_record["api_stats"] = self.client.stats

        _write_text(os.path.join(out_dir, "optimized_prompt.txt"), current_prompt)
        _write_json(os.path.join(out_dir, "optimization_log.json"), log_record)

        logger.info("  all artefacts saved to %s", out_dir)

        return {
            "dataset": dataset_name,
            "final_score": test_summary["avg_score"],
            "iterations": len(log_record["iterations"]),
            "prompt_path": os.path.join(out_dir, "optimized_prompt.txt"),
        }

    # ================================================================ #
    # Internal helpers                                                  #
    # ================================================================ #

    def _evaluate_in_chunks(
        self,
        originals: list[str],
        refinements: list[str],
        chunk_size: int = 5,
        concurrency: int | None = None,
    ) -> list[dict]:
        """Evaluate in chunks of *chunk_size*, chunks run concurrently."""
        chunks = []
        for i in range(0, len(originals), chunk_size):
            chunks.append((i, originals[i:i+chunk_size], refinements[i:i+chunk_size]))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_evals: list[dict] = [None] * len(chunks)

        def _eval_chunk(idx, o_chunk, r_chunk):
            evals = self.evaluator.evaluate_batch(o_chunk, r_chunk)
            return idx, evals

        with ThreadPoolExecutor(max_workers=concurrency or cfg.API_CONCURRENCY) as pool:
            futs = {
                pool.submit(_eval_chunk, ci, oc, rc): ci
                for ci, (_, oc, rc) in enumerate(chunks)
            }
            for fut in as_completed(futs):
                ci = futs[fut]
                try:
                    _, evals = fut.result()
                    all_evals[ci] = evals
                except Exception as exc:
                    logger.error("  eval chunk %d failed: %s", ci, exc)
                    all_evals[ci] = self.evaluator._defaults(len(chunks[ci][1]))

        flat: list[dict] = []
        for ci, (offset, _, _) in enumerate(chunks):
            for ev in (all_evals[ci] or []):
                ev["sample_id"] = ev.get("sample_id", 0) + offset
                flat.append(ev)
        return flat

    @staticmethod
    def _exclude_failures(originals: list[str], refinements: list[str]) -> tuple[list[str], list[str]]:
        """Remove API-failed pairs. Returns (filtered_originals, filtered_refinements)."""
        ok = [(o, r) for o, r in zip(originals, refinements) if r != "[REFINEMENT_ERROR]"]
        n_dropped = len(originals) - len(ok)
        if n_dropped:
            logger.warning("    excluded %d/%d API-failed samples", n_dropped, len(originals))
        if not ok:
            return [], []
        return [o for o, _ in ok], [r for _, r in ok]

    def _refine_batch(self, texts: list[str], prompt: str, concurrency: int | None = None) -> list[str]:
        """Refine texts concurrently via call_batch."""
        tasks = [
            {"system_prompt": prompt, "user_content": t}
            for t in texts
        ]
        results = self.client.call_batch(tasks, max_workers=concurrency)
        logger.info("    refined %d/%d OK", sum(1 for r in results if r != "[REFINEMENT_ERROR]"), len(texts))
        return results

    # ------------------------------------------------------------------
    def _profile_dataset(self, dataset_name: str, samples: list[str]) -> dict:
        """Profile the dataset in 4 calls: 3 batches of 5 full texts, then merge."""
        step = max(1, len(samples) // 15)
        chosen = [samples[i] for i in range(0, len(samples), step)][:15]

        batches = [chosen[0:5], chosen[5:10], chosen[10:15]]
        profile_sys = PROFILE_SYSTEM_PROMPT.format(dataset_name=dataset_name)

        tasks = []
        for batch in batches:
            if not batch:
                continue
            user_msg = "\n\n---\n\n".join(
                f"[Sample {i}]\n{t}" for i, t in enumerate(batch)
            )
            tasks.append({
                "system_prompt": profile_sys,
                "user_content": user_msg,
                "model": cfg.JUDGE_MODEL,
                "max_tokens": 2048,
            })

        results = self.client.call_batch(tasks, max_workers=len(tasks))
        partial_profiles = [r for r in results if r != "[REFINEMENT_ERROR]"]
        logger.info("    profile: %d/3 batches done", len(partial_profiles))

        if not partial_profiles:
            return {"error": "profiling_failed", "dataset_name": dataset_name}

        merge_prompt = (
            f'You are a data-quality analyst. Below are {len(partial_profiles)} partial '
            f'profiles of the "{dataset_name}" dataset. Merge them into ONE unified JSON '
            f'profile with these fields (no markdown fences):\n'
            f'{{"dominant_language","content_types","common_noise_patterns",'
            f'"has_code_blocks","has_tables","has_math","avg_quality_impression","special_notes"}}\n'
            f'Output ONLY the merged JSON object.'
        )
        merge_user = "\n\n---\n\n".join(
            f"[Partial profile {i+1}]\n{p}" for i, p in enumerate(partial_profiles)
        )
        try:
            raw = self.client.call(
                system_prompt=merge_prompt,
                user_content=merge_user,
                model=cfg.JUDGE_MODEL,
                max_tokens=2048,
            )
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                logger.info("    profile merge done")
                return json.loads(m.group())
        except Exception as exc:
            logger.warning("Profile merge failed: %s", exc)

        return {"error": "profiling_failed", "dataset_name": dataset_name}

    # ------------------------------------------------------------------
    def _meta_optimize(
        self,
        dataset_name: str,
        dataset_profile: dict,
        current_prompt: str,
        problems: list[tuple[str, str, dict]],
        summary: dict,
        iteration: int,
    ) -> str:
        examples_block = self._format_examples(problems[:5])
        reference_block = self._format_references()

        system = META_SYSTEM_TEMPLATE.format(
            dataset_name=dataset_name,
            dataset_profile=json.dumps(dataset_profile, ensure_ascii=False, indent=2),
            iteration=iteration + 1,
            max_iter=cfg.MAX_ITERATIONS,
            current_prompt=current_prompt,
            avg_score=summary["avg_score"],
            major_count=summary["major_count"],
            total_samples=summary["total_samples"],
            top_issues=json.dumps(summary["top_issues"][:5], ensure_ascii=False),
            examples_block=examples_block,
            reference_block=reference_block,
        )

        try:
            resp = self.client.call(
                system_prompt=system,
                user_content="Analyze the issues and produce an improved prompt.",
                model=cfg.JUDGE_MODEL,
            )
            return self._extract_improved_prompt(resp)
        except Exception as exc:
            logger.error("Meta-optimization call failed: %s", exc)
            return current_prompt

    # ------------------------------------------------------------------
    def _extract_improved_prompt(self, text: str) -> str:
        m = re.search(
            r"<IMPROVED_PROMPT>(.*?)</IMPROVED_PROMPT>", text, re.DOTALL
        )
        if m:
            return m.group(1).strip()
        logger.warning("No <IMPROVED_PROMPT> tags found; using full response")
        return text.strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _format_examples(problems: list[tuple[str, str, dict]]) -> str:
        parts = []
        for i, (orig, refined, ev) in enumerate(problems):
            issues = ev.get("issues", [])
            score = ev.get("total_score", "?")
            parts.append(
                f"### Problematic sample {i+1}  (score {score}/10)\n"
                f"**Original:**\n{orig}\n\n"
                f"**Model output:**\n{refined}\n\n"
                f"**Issues:** {', '.join(issues) if issues else 'none'}\n"
            )
        return "\n".join(parts) if parts else "(all samples scored well)"

    def _format_references(self) -> str:
        parts = []
        for name, content in self.reference_prompts.items():
            if name == "refinement_prompt_en.txt":
                continue
            parts.append(f"#### {name}\n{content}\n")
        return "\n".join(parts) if parts else "(no additional references)"

    # ------------------------------------------------------------------
    @staticmethod
    def _save_raw_samples(
        samples_dir: str,
        opt_set: list[str],
        test_set: list[str],
    ):
        """Persist every sampled text so they can be reproduced / audited."""
        _write_json(
            os.path.join(samples_dir, "optimization_set.json"),
            [{"id": i, "text": t} for i, t in enumerate(opt_set)],
        )
        _write_json(
            os.path.join(samples_dir, "test_set.json"),
            [{"id": i, "text": t} for i, t in enumerate(test_set)],
        )

    @staticmethod
    def _save_iteration_pairs(
        iter_dir: str,
        prefix: str,
        originals: list[str],
        refinements: list[str],
    ):
        """Save every (original, refined) pair for an iteration — full text."""
        records = [
            {"id": i, "original": o, "refined": r}
            for i, (o, r) in enumerate(zip(originals, refinements))
        ]
        _write_json(os.path.join(iter_dir, f"{prefix}_pairs.json"), records)
