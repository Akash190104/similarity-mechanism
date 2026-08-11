"""Humanity's Last Exam (HLE) benchmark.

Evaluates LLMs on 2,500 expert-level questions across dozens of academic
subjects. Supports both multiple-choice and short-answer (exact-match)
questions. Short-answer correctness is judged by an LLM (Claude Haiku).

Reference: https://arxiv.org/abs/2501.14249
Dataset:   https://huggingface.co/datasets/cais/hle
"""

import copy
import re
from typing import Any, override

from datasets import load_dataset

from benchmarks.base import Benchmark, BenchmarkResult
from src.registry.agent_registry import create_agent
from src.utils.concurrency import run_tasks

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

HLE_PROMPT = """\
{question_text}

Your response should be in the following format:
Explanation: {{your explanation for your answer choice}}
Answer: {{your chosen answer}}"""

JUDGE_PROMPT = """\
You are a precise answer-checking assistant. Determine whether a given answer \
is correct by comparing it to the expected answer. Accept equivalent forms \
(e.g. different notation, rounding, simplified fractions, unit variations).

Question: {question}
Expected answer: {expected}
Given answer: {given}

Is the given answer correct? Reply with exactly one word: CORRECT or INCORRECT."""

# ---------------------------------------------------------------------------
# HLE Benchmark
# ---------------------------------------------------------------------------


class HLE(Benchmark):
    """Humanity's Last Exam benchmark.

    Loads the cais/hle dataset from HuggingFace and evaluates agents on
    expert-level questions. Multiple-choice answers are compared directly;
    short-answer questions are judged by an LLM for semantic equivalence.

    Similarity is computed based on answer agreement between agents.
    """

    name = "hle"

    # Available categories for filtering
    CATEGORIES = [
        "Biology/Medicine",
        "Chemistry",
        "Computer Science/AI",
        "Engineering",
        "Humanities/Social Science",
        "Math",
        "Other",
        "Physics",
    ]

    def __init__(
        self,
        max_items: int | None = None,
        max_workers: int | None = None,
        store_traces: bool = False,
        text_only: bool = True,
        categories: list[str] | None = None,
        judge_model: str = "google/gemini-3.1-flash-lite-preview",
        judge_provider: str = "OpenRouter",
    ) -> None:
        """
        Args:
            max_items: Limit number of questions (for cheaper runs).
            max_workers: Max concurrent LLM calls. None = one per question.
            store_traces: If True, store full LLM response in raw_responses.
            text_only: If True, skip questions that require images.
            categories: If set, only include questions from these categories.
            judge_model: Model to use for judging short-answer correctness.
            judge_provider: Provider for the judge model.
        """
        dataset = load_dataset("cais/hle", split="test")

        questions = []
        for item in dataset:
            if text_only and item.get("image"):
                continue
            if categories and item.get("category") not in categories:
                continue
            questions.append(item)

        if max_items:
            from benchmarks.sampling import stratified_sample
            questions = stratified_sample(questions, "category", max_items)

        self.questions = questions
        self.max_workers = max_workers
        self.store_traces = store_traces
        self.judge_model = judge_model
        self.judge_provider = judge_provider

    # -- answer parsing ------------------------------------------------------

    @staticmethod
    def _parse_answer(response: str) -> str:
        """Extract the answer from the model response.

        Looks for "Answer: ..." pattern. Falls back to the last line.
        """
        match = re.search(r"Answer:\s*(.+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: last non-empty line
        lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize an answer string for comparison."""
        text = text.lower().strip()
        # Remove trailing punctuation
        text = re.sub(r"[.\s]+$", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _is_mc(question: dict) -> bool:
        """Check if a question is multiple-choice based on answer_type."""
        return question.get("answer_type", "").lower() in (
            "multiple choice",
            "multiple_choice",
            "multiplechoice",
            "mc",
        )

    def _check_mc_correct(self, model_answer: str, expected: str) -> bool:
        """Check if a multiple-choice answer is correct."""
        # Extract just the letter from model answer (e.g. "A" from "A) ...")
        model_letter = re.match(r"^\s*([A-Za-z])", model_answer)
        expected_letter = re.match(r"^\s*([A-Za-z])", expected)

        if model_letter and expected_letter:
            return model_letter.group(1).upper() == expected_letter.group(1).upper()

        # Fall back to normalized comparison
        return self._normalize(model_answer) == self._normalize(expected)

    # -- LLM judge for short-answer -----------------------------------------

    def _create_judge(self) -> Any:
        """Create an agent for judging short-answer correctness."""
        judge_config = {
            "type": "IOAgent",
            "llm": {
                "model": self.judge_model,
                "provider": self.judge_provider,
                "kwargs": {"temperature": 0},
            },
            "player_id": 99,
            "log_file": "hle_judge_log.txt",
        }
        return create_agent(judge_config)

    def _judge_answer(
        self,
        judge: Any,
        question_text: str,
        expected: str,
        given: str,
    ) -> bool:
        """Use LLM judge to determine if a short answer is correct."""
        prompt = JUDGE_PROMPT.format(
            question=question_text,
            expected=expected,
            given=given,
        )
        response, _ = judge.invoke(prompt)
        return "CORRECT" in response.upper() and "INCORRECT" not in response.upper()

    # -- main run ------------------------------------------------------------

    @override
    def run(self, agent: Any) -> BenchmarkResult:
        """Run HLE benchmark on an agent.

        Args:
            agent: An Agent instance with an `invoke(prompt)` method.

        Returns:
            BenchmarkResult with accuracy scores and per-question details.
        """
        # Phase 1: Get model responses for all questions.
        # Empty / failed responses (after the LLM client's own retry budget)
        # get a sentinel placeholder so the raw_responses display is clear and
        # the judge / MC checker reliably marks them incorrect.
        UNAVAILABLE = "Reasoning for this question is unavailable"

        def _call(q: dict) -> tuple[str, str, str]:
            prompt = HLE_PROMPT.format(question_text=q["question"])
            response, trace_id = agent.invoke(prompt)
            if not response or not response.strip():
                response = UNAVAILABLE
            parsed = self._parse_answer(response)
            return response, trace_id, parsed

        call_results = run_tasks(
            self.questions,
            _call,
            max_workers=self.max_workers,
            desc=f"hle [{agent.name.split('/')[-1][:30]}]",
        )

        # Phase 2: Judge short-answer questions
        judge = self._create_judge()
        sa_items = []  # (index, question, expected, given)
        for idx, (q, (_, _, parsed)) in enumerate(zip(self.questions, call_results)):
            if not self._is_mc(q):
                sa_items.append((idx, q, q["answer"], parsed))

        def _judge(item: tuple) -> tuple[int, bool]:
            idx, q, expected, given = item
            # Quick exact match first
            if self._normalize(given) == self._normalize(expected):
                return idx, True
            correct = self._judge_answer(judge, q["question"], expected, given)
            return idx, correct

        judge_results_list = run_tasks(
            sa_items,
            _judge,
            max_workers=self.max_workers,
            desc=f"hle-judge [{agent.name.split('/')[-1][:20]}]",
        )
        sa_verdicts = dict(judge_results_list)

        # Phase 3: Aggregate results
        raw_responses: list[dict] = []
        answer_vector: dict[str, str] = {}
        mc_correct = 0
        mc_total = 0
        sa_correct = 0
        sa_total = 0
        category_stats: dict[str, dict[str, int]] = {}

        for idx, (q, (response, trace_id, parsed)) in enumerate(
            zip(self.questions, call_results)
        ):
            qid = q["id"]
            answer_vector[qid] = parsed
            is_mc = self._is_mc(q)
            category = q.get("category", "unknown")

            if is_mc:
                mc_total += 1
                correct = self._check_mc_correct(parsed, q["answer"])
                if correct:
                    mc_correct += 1
            else:
                sa_total += 1
                correct = sa_verdicts.get(idx, False)
                if correct:
                    sa_correct += 1

            # Track per-category
            cat = category_stats.setdefault(category, {"correct": 0, "total": 0})
            cat["total"] += 1
            if correct:
                cat["correct"] += 1

            entry: dict[str, Any] = {
                "qid": qid,
                "question_text": q["question"],
                "answer_type": q.get("answer_type", ""),
                "category": category,
                "expected_answer": q["answer"],
                "chosen_answer_text": parsed,
                "correct": correct,
                "trace_id": trace_id,
            }
            if self.store_traces:
                entry["full_response"] = response

            raw_responses.append(entry)

        total = mc_total + sa_total
        total_correct = mc_correct + sa_correct

        scores: dict[str, Any] = {
            "answer_vector": answer_vector,
            "accuracy": total_correct / total if total > 0 else 0.0,
        }
        if mc_total > 0:
            scores["mc_accuracy"] = mc_correct / mc_total
        if sa_total > 0:
            scores["sa_accuracy"] = sa_correct / sa_total
        scores["by_category"] = {
            cat: stats["correct"] / stats["total"]
            for cat, stats in category_stats.items()
            if stats["total"] > 0
        }

        return BenchmarkResult(
            agent_name=agent.name,
            benchmark_name=self.name,
            scores=scores,
            raw_responses=raw_responses,
        )

    # -- similarity ----------------------------------------------------------

    @override
    def compute_similarity(
        self, result_a: BenchmarkResult, result_b: BenchmarkResult
    ) -> float:
        """Compute similarity as raw answer agreement percentage.

        Uses normalized comparison for short-answer questions.

        Returns:
            Similarity score in [0, 100] where 100 = perfect agreement.
        """
        vec_a = result_a.scores.get("answer_vector", {})
        vec_b = result_b.scores.get("answer_vector", {})

        common_qids = set(vec_a.keys()) & set(vec_b.keys())
        if not common_qids:
            return 0.0

        matches = sum(
            1 for qid in common_qids
            if self._normalize(vec_a[qid]) == self._normalize(vec_b[qid])
        )
        return matches / len(common_qids) * 100
