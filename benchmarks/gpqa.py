"""GPQA Diamond benchmark.

Graduate-level multiple-choice Q&A benchmark with 198 expert-validated
questions in biology, physics, and chemistry. All questions have 4 options.

Reference: https://arxiv.org/abs/2311.12022
Dataset:   https://huggingface.co/datasets/Idavidrein/gpqa
"""

import random
import re
from typing import Any, override

from datasets import load_dataset

from benchmarks.base import Benchmark, BenchmarkResult
from src.utils.concurrency import run_tasks

ALPHABET = "ABCD"

GPQA_PROMPT = """\
{question_text}

{answer_list}

Your response should be in the following format:
Explanation: {{your explanation for your answer choice}}
Answer: {{the letter of your chosen answer}}"""


class GPQADiamond(Benchmark):
    """GPQA Diamond benchmark.

    198 graduate-level multiple-choice questions validated by domain experts.
    All questions have exactly 4 answer options. Answers are shuffled to
    prevent position bias.

    Similarity is computed based on answer agreement between agents.
    """

    name = "gpqa"

    def __init__(
        self,
        max_items: int | None = None,
        max_workers: int | None = None,
        shuffle_answers: bool = True,
        store_traces: bool = False,
    ) -> None:
        """
        Args:
            max_items: Limit number of questions (for cheaper runs).
            max_workers: Max concurrent LLM calls. None = one per question.
            shuffle_answers: Whether to shuffle answer order per question.
            store_traces: If True, store full LLM response in raw_responses.
        """
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

        questions = []
        for idx, item in enumerate(dataset):
            questions.append({
                "id": str(idx),
                "question": item["Question"],
                "correct_answer": item["Correct Answer"],
                "incorrect_answers": [
                    item["Incorrect Answer 1"],
                    item["Incorrect Answer 2"],
                    item["Incorrect Answer 3"],
                ],
                "subdomain": item.get("Subdomain", "unknown"),
            })

        if max_items:
            from benchmarks.sampling import stratified_sample
            questions = stratified_sample(questions, "subdomain", max_items)

        self.questions = questions
        self.shuffle_answers = shuffle_answers
        self.max_workers = max_workers
        self.store_traces = store_traces

    def _prepare_question(self, q: dict) -> tuple[str, int, list[str]]:
        """Build prompt and track correct answer index.

        Returns (prompt, correct_index, answer_texts).
        """
        answers = [q["correct_answer"]] + q["incorrect_answers"]
        correct_idx = 0

        if self.shuffle_answers:
            perm = list(range(4))
            random.shuffle(perm)
            answers = [answers[i] for i in perm]
            correct_idx = perm.index(0)

        answer_list = "\n".join(
            f"{ALPHABET[i]}) {ans}" for i, ans in enumerate(answers)
        )
        prompt = GPQA_PROMPT.format(
            question_text=q["question"],
            answer_list=answer_list,
        )
        return prompt, correct_idx, answers

    @staticmethod
    def _parse_answer(response: str) -> int | None:
        """Extract the chosen answer letter from the response.

        Looks for "Answer: X" pattern. Falls back to last standalone letter.
        """
        match = re.search(r"Answer:\s*([A-Da-d])", response)
        if match:
            return ALPHABET.index(match.group(1).upper())

        # Fallback: last standalone A-D letter
        letters = re.findall(r"\b([A-D])\b", response)
        for letter in reversed(letters):
            return ALPHABET.index(letter)

        return None

    @override
    def run(self, agent: Any) -> BenchmarkResult:
        """Run GPQA Diamond benchmark on an agent."""
        prepared: list[tuple[dict, str, int, list[str]]] = []
        for q in self.questions:
            prompt, correct_idx, answers = self._prepare_question(q)
            prepared.append((q, prompt, correct_idx, answers))

        def _call(item: tuple) -> tuple[str, str, int | None]:
            _, prompt, _, _ = item
            response, trace_id = agent.invoke(prompt)
            chosen = self._parse_answer(response)
            return response, trace_id, chosen

        call_results = run_tasks(
            prepared,
            _call,
            max_workers=self.max_workers,
            desc=f"gpqa [{agent.name.split('/')[-1][:30]}]",
        )

        raw_responses: list[dict] = []
        answer_vector: dict[str, int | None] = {}
        total_correct = 0
        subdomain_stats: dict[str, dict[str, int]] = {}

        for (q, _, correct_idx, answers), (response, trace_id, chosen) in zip(
            prepared, call_results
        ):
            qid = q["id"]
            answer_vector[qid] = chosen
            correct = chosen == correct_idx
            if correct:
                total_correct += 1

            subdomain = q["subdomain"]
            stats = subdomain_stats.setdefault(subdomain, {"correct": 0, "total": 0})
            stats["total"] += 1
            if correct:
                stats["correct"] += 1

            entry: dict[str, Any] = {
                "qid": qid,
                "question_text": q["question"],
                "subdomain": subdomain,
                "answer_options": [
                    f"{ALPHABET[i]}) {ans}" for i, ans in enumerate(answers)
                ],
                "correct_index": correct_idx,
                "correct_letter": ALPHABET[correct_idx],
                "chosen_index": chosen,
                "chosen_letter": ALPHABET[chosen] if chosen is not None else None,
                "chosen_answer_text": answers[chosen] if chosen is not None else None,
                "correct": correct,
                "trace_id": trace_id,
            }
            if self.store_traces:
                entry["full_response"] = response

            raw_responses.append(entry)

        total = len(self.questions)
        scores: dict[str, Any] = {
            "answer_vector": answer_vector,
            "accuracy": total_correct / total if total > 0 else 0.0,
            "by_subdomain": {
                sub: stats["correct"] / stats["total"]
                for sub, stats in subdomain_stats.items()
                if stats["total"] > 0
            },
        }

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
        """Compute similarity using Cohen's Kappa (chance-corrected agreement).

        All GPQA questions have 4 options, so p_chance = 0.25.

        Returns:
            Similarity score in [0, 100] where 50 = chance-level agreement.
        """
        vec_a = result_a.scores.get("answer_vector", {})
        vec_b = result_b.scores.get("answer_vector", {})

        common_qids = set(vec_a.keys()) & set(vec_b.keys())
        if not common_qids:
            return 50.0

        matches = sum(1 for qid in common_qids if vec_a[qid] == vec_b[qid])
        p_o = matches / len(common_qids)
        p_e = 1 / 4  # all questions have 4 options

        kappa = (p_o - p_e) / (1 - p_e)
        return (kappa + 1) / 2 * 100
