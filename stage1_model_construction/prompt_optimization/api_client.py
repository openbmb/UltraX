"""
Unified API client with concurrency support.

- REFINE_MODEL  → End-to-end data refinement
- JUDGE_MODEL   → Evaluation / Meta-Optimizer / Dataset profiling
"""

import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config as cfg

logger = logging.getLogger(__name__)


class LLMClient:
    """Thread-safe wrapper with built-in concurrency pool."""

    def __init__(self, api_key: str | None = None):
        self.url = cfg.API_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or cfg.API_KEY}",
        }
        self._lock = threading.Lock()
        self._total_calls = 0
        self._total_tokens_est = 0

    # ------------------------------------------------------------------
    def call(
        self,
        system_prompt: str,
        user_content: str = "",
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Single chat completion request (thread-safe)."""
        messages = [{"role": "system", "content": system_prompt}]
        if user_content:
            messages.append({"role": "user", "content": user_content})

        resolved_model = model or cfg.REFINE_MODEL
        if temperature is not None:
            resolved_temp = temperature
        elif resolved_model == cfg.REFINE_MODEL:
            resolved_temp = cfg.REFINE_TEMPERATURE
        else:
            resolved_temp = cfg.JUDGE_TEMPERATURE

        body = {
            "model": resolved_model,
            "messages": messages,
            "temperature": resolved_temp,
            "max_tokens": max_tokens or cfg.API_MAX_TOKENS,
            "enable_thinking": False,
        }

        last_exc = None
        for attempt in range(1, cfg.API_RETRY_TIMES + 1):
            try:
                resp = requests.post(
                    self.url,
                    json=body,
                    headers=self.headers,
                    timeout=cfg.API_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()

                content = data["choices"][0]["message"]["content"]

                with self._lock:
                    self._total_calls += 1
                    usage = data.get("usage", {})
                    if usage:
                        self._total_tokens_est += usage.get("total_tokens", 0)

                return content

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "API [%s] attempt %d/%d failed: %s",
                    resolved_model,
                    attempt,
                    cfg.API_RETRY_TIMES,
                    exc,
                )
                if attempt < cfg.API_RETRY_TIMES:
                    time.sleep(cfg.API_RETRY_DELAY * attempt)

        raise RuntimeError(
            f"API call failed after {cfg.API_RETRY_TIMES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    def call_batch(
        self,
        tasks: list[dict],
        max_workers: int | None = None,
    ) -> list[str]:
        """
        Concurrently execute multiple call() requests.

        Parameters
        ----------
        tasks : list of dict
            Each dict is passed as kwargs to call(), e.g.
            {"system_prompt": "...", "user_content": "...", "model": "..."}
        max_workers : int, optional
            Concurrency limit.  Defaults to cfg.API_CONCURRENCY.

        Returns
        -------
        list[str]  — results in the SAME order as *tasks*.
        """
        workers = max_workers or cfg.API_CONCURRENCY
        results: list[str] = ["[REFINEMENT_ERROR]"] * len(tasks)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {}
            for idx, task in enumerate(tasks):
                try:
                    future_to_idx[pool.submit(self.call, **task)] = idx
                except Exception as exc:
                    logger.error("Failed to submit batch item %d: %s", idx, exc)

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Batch item %d failed: %s", idx, exc)

        return results

    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "total_tokens_est": self._total_tokens_est,
            }
