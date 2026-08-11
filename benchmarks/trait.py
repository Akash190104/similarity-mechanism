"""TRAIT personality benchmark.

Evaluates LLMs on ~8,000 situational personality questions across 8 traits
(Big Five + Dark Triad). Each question has 4 options: 2 high-trait, 2 low-trait.
No correct answer — measures personality expression levels.

Traits: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism,
        Machiavellianism, Narcissism, Psychopathy

Reference: https://arxiv.org/abs/2406.14703
Dataset:   https://huggingface.co/datasets/mirlab/TRAIT
"""

import random
import re
from typing import Any, override

from datasets import load_dataset

from benchmarks.base import Benchmark, BenchmarkResult
from src.utils.concurrency import run_tasks

ALPHABET = "ABCD"

ALL_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
]

TRAIT_PROMPT = """\
{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Your response should be in the following format:
Explanation: {{your explanation for your choice}}
Answer: {{the letter of your chosen answer}}"""


class TRAIT(Benchmark):
    """TRAIT personality benchmark.

    ~8,000 situational MCQ questions across 8 personality traits.
    Each question has 4 options (2 high-trait, 2 low-trait). Options are
    shuffled to prevent position bias. Measures personality expression.
    Similarity is computed based on choice agreement.
    """

    name = "trait"

    def __init__(
        self,
        traits: list[str] | None = None,
        max_items: int | None = None,
        max_workers: int | None = None,
        shuffle_answers: bool = True,
        store_traces: bool = False,
    ) -> None:
        """
        Args:
            traits: List of trait names to include. None = all 8 traits.
            max_items: Limit number of questions (for cheaper runs).
            max_workers: Max concurrent LLM calls. None = one per question.
            shuffle_answers: Whether to shuffle answer order per question.
            store_traces: If True, store full LLM response in raw_responses.
        """
        selected_traits = traits or ALL_TRAITS

        questions = []
        for trait in selected_traits:
            split_data = load_dataset("mirlab/TRAIT", split=trait)
            for idx, item in enumerate(split_data):
                questions.append({
                    "id": f"{trait}_{idx}",
                    "trait": trait,
                    "question": item["question"],
                    "high1": item["response_high1"],
                    "high2": item["response_high2"],
                    "low1": item["response_low1"],
                    "low2": item["response_low2"],
                })

        if max_items:
            from benchmarks.sampling import stratified_sample
            questions = stratified_sample(questions, "trait", max_items)

        self.questions = questions
        self.shuffle_answers = shuffle_answers
        self.max_workers = max_workers
        self.store_traces = store_traces

    def _prepare_question(self, q: dict) -> tuple[str, list[bool]]:
        """Build prompt and track which options are high-trait.

        Returns (prompt, is_high) where is_high[i] is True if option i
        is a high-trait response.
        """
        options = [
            (q["high1"], True),
            (q["high2"], True),
            (q["low1"], False),
            (q["low2"], False),
        ]

        if self.shuffle_answers:
            random.shuffle(options)

        texts = [o[0] for o in options]
        is_high = [o[1] for o in options]

        prompt = TRAIT_PROMPT.format(
            question=q["question"],
            option_a=texts[0],
            option_b=texts[1],
            option_c=texts[2],
            option_d=texts[3],
        )
        return prompt, is_high

    @staticmethod
    def _parse_answer(response: str) -> int | None:
        """Extract the chosen answer index (0-3) from the response."""
        match = re.search(r"Answer:\s*([A-Da-d])", response)
        if match:
            return ALPHABET.index(match.group(1).upper())

        letters = re.findall(r"\b([A-D])\b", response)
        for letter in reversed(letters):
            return ALPHABET.index(letter)

        return None

    @override
    def run(self, agent: Any) -> BenchmarkResult:
        """Run TRAIT benchmark on an agent."""
        prepared: list[tuple[dict, str, list[bool]]] = []
        for q in self.questions:
            prompt, is_high = self._prepare_question(q)
            prepared.append((q, prompt, is_high))

        def _call(item: tuple) -> tuple[str, str, int | None]:
            _, prompt, _ = item
            response, trace_id = agent.invoke(prompt)
            chosen = self._parse_answer(response)
            return response, trace_id, chosen

        call_results = run_tasks(
            prepared,
            _call,
            max_workers=self.max_workers,
            desc=f"trait [{agent.name.split('/')[-1][:25]}]",
        )

        raw_responses: list[dict] = []
        answer_vector: dict[str, str | None] = {}
        trait_high_counts: dict[str, int] = {}
        trait_totals: dict[str, int] = {}

        for (q, _, is_high), (response, trace_id, chosen) in zip(
            prepared, call_results
        ):
            qid = q["id"]
            chosen_letter = ALPHABET[chosen] if chosen is not None else None
            answer_vector[qid] = chosen_letter

            trait = q["trait"]
            chose_high = None
            if chosen is not None:
                chose_high = is_high[chosen]
                trait_totals[trait] = trait_totals.get(trait, 0) + 1
                if chose_high:
                    trait_high_counts[trait] = trait_high_counts.get(trait, 0) + 1

            entry: dict[str, Any] = {
                "qid": qid,
                "trait": trait,
                "question_text": q["question"],
                "chosen_letter": chosen_letter,
                "chose_high_trait": chose_high,
                "trace_id": trace_id,
            }
            if self.store_traces:
                entry["full_response"] = response

            raw_responses.append(entry)

        trait_scores = {
            trait: trait_high_counts.get(trait, 0) / trait_totals[trait]
            for trait in trait_totals
            if trait_totals[trait] > 0
        }

        scores: dict[str, Any] = {
            "answer_vector": answer_vector,
            "trait_scores": trait_scores,
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
        """Compute similarity as raw answer agreement percentage.

        Returns:
            Similarity score in [0, 100] where 100 = perfect agreement.
        """
        vec_a = result_a.scores.get("answer_vector", {})
        vec_b = result_b.scores.get("answer_vector", {})

        common = set(vec_a.keys()) & set(vec_b.keys())
        if not common:
            return 0.0

        matches = sum(1 for qid in common if vec_a[qid] == vec_b[qid])
        return matches / len(common) * 100
