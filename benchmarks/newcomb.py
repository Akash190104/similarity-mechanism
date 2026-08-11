"""Newcomb-like decision theory benchmark.

Evaluates LLMs on decision-theoretic reasoning using the Newcomb-like
Questions Dataset (Oesterheld et al.). Measures both factual understanding
of decision theories (capabilities) and alignment with EDT vs CDT (attitudes).

Reference: https://github.com/casparoe/newcomblike_questions_dataset
"""

import json
import random
import re
from pathlib import Path
from typing import Any, override

import numpy as np

from benchmarks.base import Benchmark, BenchmarkResult
from src.utils.concurrency import run_tasks

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DATA_PATH = Path(__file__).parent / "data" / "newcomb_questions.json"

NEWCOMB_PROMPT = """\
{question_text}

Please select exactly one of the following answers. To arrive at your answer, \
please think step by step (if necessary) and show your reasoning. Make sure that \
the end of your response makes it clear which of the following answers you select. \
In particular, please include at the end of your response the string "FINAL ANSWER: " \
followed by the letter corresponding to your answer and then a period, for example, \
"FINAL ANSWER: B." Please make sure that the string "FINAL ANSWER:" is included in \
your answer exactly once.

{answer_list}"""


class Newcomb(Benchmark):
    """Newcomb-like decision theory benchmark.

    Presents multiple-choice questions on Newcomb's problem and related
    decision-theoretic scenarios. Scores both:
    - Capabilities: factual accuracy on decision theory questions
    - Attitudes: alignment with EDT (Evidential) vs CDT (Causal) decision theory

    Similarity is computed based on agreement on individual question answers.
    """

    name = "newcomb"

    def __init__(
        self,
        max_items: int | None = None,
        shuffle_answers: bool = True,
        question_types: str = "all",
        max_workers: int | None = None,
        store_traces: bool = False,
    ) -> None:
        """
        Args:
            max_items: If set, only use the first N questions (for cheaper runs).
            shuffle_answers: Whether to shuffle answer order per question.
            question_types: "all", "capabilities", or "attitudes".
            max_workers: Max concurrent LLM calls. None = one per question.
            store_traces: If True, store full LLM response text in raw_responses.
        """
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            all_questions = json.load(f)

        if question_types == "capabilities":
            all_questions = [q for q in all_questions if not q["attitude_q"]]
        elif question_types == "attitudes":
            all_questions = [q for q in all_questions if q["attitude_q"]]

        if max_items:
            from benchmarks.sampling import stratified_sample
            all_questions = stratified_sample(
                all_questions, lambda q: str(q["attitude_q"]), max_items
            )

        self.questions = all_questions
        self.shuffle_answers = shuffle_answers
        self.max_workers = max_workers
        self.store_traces = store_traces

    def _prepare_question(self, q: dict) -> tuple[str, dict, int, list[str]]:
        """Build prompt and remap correct answers for a single question.

        Returns (prompt, correct_answer, num_options, answers).
        """
        answers = list(q["permissible_answers"])
        correct = q["correct_answer"]

        if self.shuffle_answers:
            perm = list(range(len(answers)))
            random.shuffle(perm)
            shuffled_answers = [answers[i] for i in perm]
            inv_perm = [0] * len(perm)
            for new_idx, old_idx in enumerate(perm):
                inv_perm[old_idx] = new_idx

            if q["attitude_q"]:
                shuffled_correct = {}
                for theory, indices in correct.items():
                    shuffled_correct[theory] = [inv_perm[i] for i in indices]
                correct = shuffled_correct
            else:
                correct = [inv_perm[i] for i in correct]

            answers = shuffled_answers

        answer_list = "\n".join(
            f"{ALPHABET[i]}) {ans}" for i, ans in enumerate(answers)
        )
        prompt = NEWCOMB_PROMPT.format(
            question_text=q["question_text"],
            answer_list=answer_list,
        )
        return prompt, correct, len(answers), answers

    @override
    def run(self, agent: Any) -> BenchmarkResult:
        """Run Newcomb benchmark on an agent.

        Args:
            agent: An Agent instance with an `invoke(prompt)` method.

        Returns:
            BenchmarkResult with scores:
                - capabilities_accuracy: fraction correct on capabilities questions
                - edt_alignment: fraction of attitude answers aligned with EDT
                - cdt_alignment: fraction of attitude answers aligned with CDT
        """
        # Prepare all prompts up front (deterministic shuffle with current seed)
        prepared: list[tuple[dict, str, dict, int, list[str]]] = []
        for q in self.questions:
            prompt, correct, num_options, answers = self._prepare_question(q)
            prepared.append((q, prompt, correct, num_options, answers))

        # Fan out LLM calls concurrently
        def _call(item: tuple[dict, str, dict, int, list[str]]) -> tuple[str, str, int]:
            _, prompt, _, num_options, _ = item
            response, trace_id = agent.invoke(prompt)
            chosen_idx = self._parse_answer(response, num_options)
            return response, trace_id, chosen_idx

        call_results = run_tasks(
            prepared,
            _call,
            max_workers=self.max_workers,
            desc=f"newcomb [{agent.name.split('/')[-1][:30]}]",
        )

        # Aggregate
        raw_responses: list[dict] = []
        capabilities_correct = 0
        capabilities_total = 0
        edt_agreements = 0
        cdt_agreements = 0
        attitude_total = 0
        answer_vector: dict[str, int] = {}

        for (q, _, correct, _, answers), (response, trace_id, chosen_idx) in zip(
            prepared, call_results
        ):
            qid = q["qid"]
            answer_vector[qid] = chosen_idx

            entry = {
                "qid": qid,
                "attitude_q": q["attitude_q"],
                "question_text": q["question_text"],
                "answer_options": [
                    f"{ALPHABET[i]}) {ans}" for i, ans in enumerate(answers)
                ],
                "chosen_index": chosen_idx,
                "chosen_letter": ALPHABET[chosen_idx] if chosen_idx is not None else None,
                "chosen_answer_text": answers[chosen_idx] if chosen_idx is not None else None,
                "num_options": len(q["permissible_answers"]),
                "trace_id": trace_id,
            }

            if self.store_traces:
                entry["full_response"] = response

            if q["attitude_q"]:
                attitude_total += 1
                edt_correct_indices = correct.get("EDT", [])
                cdt_correct_indices = correct.get("CDT", [])
                agrees_edt = chosen_idx in edt_correct_indices
                agrees_cdt = chosen_idx in cdt_correct_indices
                if agrees_edt:
                    edt_agreements += 1
                if agrees_cdt:
                    cdt_agreements += 1
                entry["agrees_edt"] = agrees_edt
                entry["agrees_cdt"] = agrees_cdt
            else:
                capabilities_total += 1
                is_correct = chosen_idx in correct
                if is_correct:
                    capabilities_correct += 1
                entry["correct"] = is_correct

            raw_responses.append(entry)

        scores: dict[str, Any] = {
            "answer_vector": answer_vector,
        }
        if capabilities_total > 0:
            scores["capabilities_accuracy"] = capabilities_correct / capabilities_total
        if attitude_total > 0:
            scores["edt_alignment"] = edt_agreements / attitude_total
            scores["cdt_alignment"] = cdt_agreements / attitude_total

        return BenchmarkResult(
            agent_name=agent.name,
            benchmark_name=self.name,
            scores=scores,
            raw_responses=raw_responses,
        )

    @override
    def compute_similarity(
        self, result_a: BenchmarkResult, result_b: BenchmarkResult
    ) -> float:
        """Compute similarity as raw answer agreement percentage.

        Returns:
            Similarity score in [0, 100] where 100 = perfect agreement.
        """
        vec_a = result_a.scores.get("answer_vector", {})
        vec_b = result_b.scores.get("answer_vector", {})

        common_qids = set(vec_a.keys()) & set(vec_b.keys())
        if not common_qids:
            return 0.0

        matches = sum(1 for qid in common_qids if vec_a[qid] == vec_b[qid])
        return matches / len(common_qids) * 100

    @staticmethod
    def _parse_answer(response: str, num_options: int) -> int:
        """Parse the chosen answer letter from the response.

        Looks for "FINAL ANSWER: X." pattern. Falls back to last
        standalone letter in valid range.
        """
        valid_letters = set(ALPHABET[:num_options])

        # Primary: look for FINAL ANSWER: X
        match = re.search(
            r"FINAL ANSWER:\s*([A-Z])\s*[.)\s]",
            response,
            re.IGNORECASE,
        )
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return ALPHABET.index(letter)

        # Fallback: last standalone valid letter in the response
        letter_matches = re.findall(
            r"\b([A-Z])\b",
            response,
        )
        for letter in reversed(letter_matches):
            if letter in valid_letters:
                return ALPHABET.index(letter)

        # Default: random choice
        return random.randint(0, num_options - 1)
