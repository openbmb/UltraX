"""
LLM-based quality evaluator for text refinement results.

Sends a batch of (original, refined) pairs to the judge model and
returns structured scores + issue descriptions that feed into the
meta-optimizer.
"""

import json
import re
import logging

from api_client import LLMClient
import config as cfg

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Judge system prompt                                                  #
# ------------------------------------------------------------------ #
JUDGE_SYSTEM_PROMPT = """\
You are a strict Quality Assurance Judge for a **web-text cleaning pipeline**
whose sole purpose is to improve pre-training corpus quality.

CRITICAL CONTEXT — this is a CLEANING task, NOT a rewriting task:
- The model should ONLY delete noise (ads, navigation, HTML tags, boilerplate,
  gibberish, engagement bait, contact info, etc.).
- Clean text that needs no modification MUST be output **exactly as-is**,
  character-for-character.  Any unnecessary rewriting, paraphrasing,
  summarizing, or "style improvement" is a FAILURE.
- Completely valueless content (pure ads, gibberish, spam) should be
  replaced with exactly: `[Content valueless, deleted]`

You will receive multiple (Original, Refined) text pairs.  For **each** pair,
evaluate on these five dimensions (0-2 each, 10 total):

1. **Noise Removal** (0-2)
   2 = all noise properly removed (ads, nav bars, boilerplate, HTML tags, gibberish)
   1 = some noise remains
   0 = significant noise left untouched

2. **No Over-Editing** (0-2)  ← MOST IMPORTANT
   2 = clean portions of the text are completely untouched; zero unnecessary changes
   1 = minor unnecessary edits (rephrasing, word changes, reordering)
   0 = significant rewriting, summarizing, or adding content not in the original

3. **Content Preservation** (0-2)
   2 = all valuable content kept intact (nothing useful was deleted)
   1 = minor valuable content lost
   0 = significant content wrongly removed

4. **Format Integrity** (0-2)
   2 = markdown headers, lists, tables, code blocks preserved correctly
   1 = minor format issues
   0 = structure broken or stripped

5. **Valueless Detection** (0-2)
   2 = purely valueless content correctly tagged "[Content valueless, deleted]";
       valuable content correctly kept
   1 = borderline call
   0 = wrong decision (valuable content deleted, or pure junk fully kept)

For each sample also list concrete **issues** (short phrases) and a
**severity** tag: "none", "minor", or "major".

Return **ONLY** valid JSON (no markdown fences, no explanation), schema:

{
  "evaluations": [
    {
      "sample_id": <int>,
      "scores": {
        "noise_removal": <0-2>,
        "no_over_editing": <0-2>,
        "content_preservation": <0-2>,
        "format_integrity": <0-2>,
        "valueless_detection": <0-2>
      },
      "total_score": <0-10>,
      "issues": ["issue description", ...],
      "severity": "none|minor|major"
    }
  ]
}
"""


# no truncation — full text sent to judge


# ------------------------------------------------------------------ #
# Evaluator class                                                      #
# ------------------------------------------------------------------ #
class RefinementEvaluator:
    def __init__(self, client: LLMClient):
        self.client = client

    # ---- public API ------------------------------------------------
    def evaluate_batch(
        self,
        originals: list[str],
        refinements: list[str],
    ) -> list[dict]:
        """Evaluate a batch; returns list of per-sample dicts."""
        user_msg = self._build_user_message(originals, refinements)
        try:
            raw = self.client.call(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_content=user_msg,
                model=cfg.JUDGE_MODEL,
            )
            return self._parse(raw, len(originals))
        except Exception as exc:
            logger.error("Batch evaluation failed: %s", exc)
            return self._defaults(len(originals))

    def summarize(self, evaluations: list[dict]) -> dict:
        """Aggregate per-sample evaluations into a compact summary."""
        scores = [e.get("total_score", 5) for e in evaluations]
        avg = sum(scores) / len(scores) if scores else 0.0

        severity_cnt = {"none": 0, "minor": 0, "major": 0}
        issue_freq: dict[str, int] = {}
        for e in evaluations:
            sev = e.get("severity", "minor")
            severity_cnt[sev] = severity_cnt.get(sev, 0) + 1
            for iss in e.get("issues", []):
                issue_freq[iss] = issue_freq.get(iss, 0) + 1

        return {
            "avg_score": round(avg, 2),
            "severity_distribution": severity_cnt,
            "top_issues": sorted(issue_freq.items(), key=lambda x: -x[1])[:10],
            "total_samples": len(evaluations),
            "major_count": severity_cnt.get("major", 0),
        }

    # ---- internal --------------------------------------------------
    def _build_user_message(self, originals, refinements):
        parts = []
        for i, (o, r) in enumerate(zip(originals, refinements)):
            parts.append(
                f"=== Sample {i} ===\n"
                f"[Original]\n{o}\n\n"
                f"[Refined]\n{r}\n"
            )
        return "\n".join(parts)

    def _parse(self, raw: str, n: int) -> list[dict]:
        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                data = json.loads(m.group())
                evals = data.get("evaluations", [])
                if len(evals) == n:
                    return evals
                logger.warning(
                    "Judge returned %d evaluations, expected %d", len(evals), n
                )
                if evals:
                    return evals
        except json.JSONDecodeError:
            logger.warning("Judge response is not valid JSON")
        return self._defaults(n)

    @staticmethod
    def _defaults(n: int) -> list[dict]:
        return [
            {
                "sample_id": i,
                "scores": {
                    "noise_removal": 1,
                    "no_over_editing": 1,
                    "content_preservation": 1,
                    "format_integrity": 1,
                    "valueless_detection": 1,
                },
                "total_score": 5,
                "issues": ["evaluation_parse_failed"],
                "severity": "minor",
            }
            for i in range(n)
        ]
