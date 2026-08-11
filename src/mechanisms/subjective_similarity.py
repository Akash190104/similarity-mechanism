"""Subjective similarity: agents judge other agents' benchmark responses.

Instead of algorithmic similarity (answer agreement %), each agent reads
the other agent's benchmark responses and rates how similar the other
agent's decision-making is to its own.

Three modes:
- "decision": Agent sees questions + chosen answers (no reasoning traces)
- "explanation": Agent sees questions + reasoning traces (final answers redacted)
- "both": Agent sees questions + reasoning traces + final answers
"""

import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmarks.base import BenchmarkResult
from benchmarks.llm_judge import LLMJudge, QAPair
from src.utils.concurrency import run_tasks

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SUBJECTIVE_PROMPT_DECISION = """\
You are about to play a strategic game against another agent. Before the \
game, both you and the other agent were independently given a set of \
questions. Below are the other agent's responses to those questions. You \
do NOT see your own responses here — only theirs.

Based on these responses, assess how similar the other agent's \
decision-making style is to your own. Consider:
- Do their answers suggest they would reach similar conclusions as you?
- Do they seem to apply similar reasoning strategies as you would?
- Do they show similar preferences or biases as you?

{dossier}

Provide a similarity score from 0 to 100, where:
- 0 means their decision-making is completely different from yours
- 50 means moderately similar to yours
- 100 means nearly identical to your decision-making style

Think step by step about what their answers reveal about their \
decision-making, compare it to how you would approach the same problems, \
and then provide your final score.

Your response MUST end with exactly: SIMILARITY SCORE: <number>"""


SUBJECTIVE_PROMPT_EXPLANATION = """\
You are about to play a strategic game against another agent. Before the \
game, both you and the other agent were independently given a set of \
questions. Below are the other agent's reasoning processes for those \
questions. Their final answers have been redacted — you can only see how \
they think, not what they concluded. You do NOT see your own responses \
here — only theirs.

Based on their reasoning, assess how similar the other agent's \
decision-making style is to your own. Consider:
- Do they follow similar chains of reasoning as you would?
- Do they weigh similar factors when making decisions?
- Do they show similar analytical approaches as you?
- Do their thought processes suggest similar biases or preferences as yours?

{dossier}

Provide a similarity score from 0 to 100, where:
- 0 means their reasoning style is completely different from yours
- 50 means moderately similar to yours
- 100 means nearly identical to your reasoning style

Think step by step about what their reasoning reveals about their \
decision-making process, compare it to how you would approach the same \
problems, and then provide your final score.

Your response MUST end with exactly: SIMILARITY SCORE: <number>"""


SUBJECTIVE_PROMPT_BOTH = """\
You are about to play a strategic game against another agent. Before the \
game, both you and the other agent were independently given a set of \
questions. Below are the other agent's reasoning processes and final \
answers to those questions. You do NOT see your own responses here — \
only theirs.

Based on their reasoning and answers, assess how similar the other \
agent's decision-making style is to your own. Consider:
- Do they follow similar chains of reasoning as you would?
- Do they reach similar conclusions as you?
- Do they weigh similar factors when making decisions?
- Do they show similar analytical approaches, preferences, or biases as you?

{dossier}

Provide a similarity score from 0 to 100, where:
- 0 means their decision-making is completely different from yours
- 50 means moderately similar to yours
- 100 means nearly identical to your decision-making style

Think step by step about what their reasoning and answers reveal about \
their decision-making, compare it to how you would approach the same \
problems, and then provide your final score.

Your response MUST end with exactly: SIMILARITY SCORE: <number>"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubjectiveJudgment:
    """Result from one agent's subjective similarity rating of another."""

    judge_name: str
    target_name: str
    similarity_score: float  # 0-100
    trace_id: str  # comma-separated if chunked
    raw_response: str
    chunk_scores: list[float] | None = None
    num_questions_shown: int = 0


# ---------------------------------------------------------------------------
# SubjectiveSimilarityComputer
# ---------------------------------------------------------------------------


class SubjectiveSimilarityComputer:
    """Compute subjective similarity: agents rate how similar other agents are.

    For each ordered pair (A, B), creates a fresh instance of A, shows it
    B's benchmark responses, and asks it to rate similarity 0-100.
    The agent does NOT see its own responses.
    """

    def __init__(
        self,
        mode: str = "decision",
        chunk_size: int | None = None,
        max_items: int | None = None,
        max_workers: int | None = None,
    ) -> None:
        """
        Args:
            mode: "decision" (show answers), "explanation" (show reasoning, redact answers),
                or "both" (show reasoning + answers).
            chunk_size: If set, split questions into chunks of this size,
                judge each chunk independently, and average.
            max_items: Max questions to show each judge. Randomly sampled if exceeded.
            max_workers: Concurrency for judge calls.
        """
        if mode not in ("decision", "explanation", "both"):
            raise ValueError(
                f"Unknown subjective mode: {mode!r}. Must be 'decision', 'explanation', or 'both'."
            )
        self.mode = mode
        self.chunk_size = chunk_size
        self.max_items = max_items
        self.max_workers = max_workers

    # -- dossier preparation -------------------------------------------------

    @staticmethod
    def _extract_target_dossier_decision(qa_pairs: list[QAPair]) -> list[dict]:
        """Format Q&A pairs for decision-based judging (questions + answers)."""
        entries = []
        for i, pair in enumerate(qa_pairs, 1):
            entry: dict[str, Any] = {
                "index": i,
                "question": pair.question_text,
                "answer": pair.chosen_answer_text,
            }
            if pair.answer_options:
                entry["options"] = pair.answer_options
            entries.append(entry)
        return entries

    @staticmethod
    def _extract_target_dossier_explanation(qa_pairs: list[QAPair]) -> list[dict]:
        """Format Q&A pairs for explanation-based judging (reasoning, answers redacted)."""
        entries = []
        for i, pair in enumerate(qa_pairs, 1):
            reasoning = pair.full_response or ""
            if reasoning:
                reasoning = LLMJudge._strip_final_answer(reasoning)
            entry: dict[str, Any] = {
                "index": i,
                "question": pair.question_text,
                "reasoning": reasoning if reasoning else "[No reasoning trace available]",
            }
            if pair.answer_options:
                entry["options"] = pair.answer_options
            entries.append(entry)
        return entries

    @staticmethod
    def _extract_target_dossier_both(qa_pairs: list[QAPair]) -> list[dict]:
        """Format Q&A pairs showing both reasoning and answers."""
        entries = []
        for i, pair in enumerate(qa_pairs, 1):
            reasoning = pair.full_response or ""
            if reasoning:
                reasoning = LLMJudge._strip_final_answer(reasoning)
            entry: dict[str, Any] = {
                "index": i,
                "question": pair.question_text,
                "reasoning": reasoning if reasoning else "[No reasoning trace available]",
                "answer": pair.chosen_answer_text,
            }
            if pair.answer_options:
                entry["options"] = pair.answer_options
            entries.append(entry)
        return entries

    def _format_dossier_text(self, entries: list[dict]) -> str:
        """Convert structured dossier entries to formatted text."""
        lines: list[str] = []
        for entry in entries:
            lines.append(f"--- Question {entry['index']} ---")
            lines.append(f"Q: {entry['question']}")
            if "options" in entry:
                lines.append("Options:")
                for opt in entry["options"]:
                    lines.append(f"  {opt}")
            if self.mode == "decision":
                lines.append(f"Their answer: {entry['answer']}")
            elif self.mode == "both":
                lines.append(f"Their reasoning:\n{entry['reasoning']}")
                lines.append(f"Their answer: {entry['answer']}")
            else:
                lines.append(f"Their reasoning:\n{entry['reasoning']}")
            lines.append("")
        return "\n".join(lines)

    def _build_judge_prompt(self, dossier_text: str) -> str:
        """Build the prompt asking the agent to rate similarity."""
        if self.mode == "decision":
            return SUBJECTIVE_PROMPT_DECISION.format(dossier=dossier_text)
        elif self.mode == "both":
            return SUBJECTIVE_PROMPT_BOTH.format(dossier=dossier_text)
        return SUBJECTIVE_PROMPT_EXPLANATION.format(dossier=dossier_text)

    # -- score parsing -------------------------------------------------------

    @staticmethod
    def _parse_score(response: str) -> int:
        """Parse similarity score from response. Reuses LLMJudge pattern."""
        return LLMJudge._parse_score(response)

    # -- single judgment -----------------------------------------------------

    def _judge_single(
        self,
        judge_agent: Any,
        target_qa: list[QAPair],
    ) -> SubjectiveJudgment:
        """Have one agent judge another agent's responses.

        If chunk_size is set, splits into chunks and averages.
        """
        pairs = list(target_qa)
        if self.max_items and len(pairs) > self.max_items:
            pairs = random.sample(pairs, self.max_items)

        if self.chunk_size and len(pairs) > self.chunk_size:
            return self._judge_single_chunked(judge_agent, pairs)

        # Build dossier
        if self.mode == "decision":
            entries = self._extract_target_dossier_decision(pairs)
        elif self.mode == "both":
            entries = self._extract_target_dossier_both(pairs)
        else:
            entries = self._extract_target_dossier_explanation(pairs)

        dossier_text = self._format_dossier_text(entries)
        prompt = self._build_judge_prompt(dossier_text)

        response, trace_id, score = judge_agent.chat_with_retries(
            prompt, self._parse_score
        )
        return SubjectiveJudgment(
            judge_name=judge_agent.name,
            target_name="",  # filled by caller
            similarity_score=float(score),
            trace_id=trace_id,
            raw_response=response,
            num_questions_shown=len(pairs),
        )

    def _judge_single_chunked(
        self,
        judge_agent: Any,
        pairs: list[QAPair],
    ) -> SubjectiveJudgment:
        """Judge in chunks and average scores."""
        assert self.chunk_size is not None
        chunks = [
            pairs[i : i + self.chunk_size]
            for i in range(0, len(pairs), self.chunk_size)
        ]
        chunk_scores: list[float] = []
        trace_ids: list[str] = []
        last_response = ""

        from tqdm import tqdm

        for chunk_idx, chunk in enumerate(
            tqdm(
                chunks,
                desc=f"  chunks [{judge_agent.name.split('/')[-1][:15]}]",
                leave=False,
            )
        ):
            if self.mode == "decision":
                entries = self._extract_target_dossier_decision(chunk)
            elif self.mode == "both":
                entries = self._extract_target_dossier_both(chunk)
            else:
                entries = self._extract_target_dossier_explanation(chunk)

            dossier_text = self._format_dossier_text(entries)
            prompt = self._build_judge_prompt(dossier_text)
            prompt += (
                f"\n\n(This is batch {chunk_idx + 1} of {len(chunks)}. "
                f"Score this batch independently.)"
            )

            response, trace_id, score = judge_agent.chat_with_retries(
                prompt, self._parse_score
            )
            chunk_scores.append(float(score))
            trace_ids.append(trace_id)
            last_response = response

        avg_score = float(np.mean(chunk_scores))
        return SubjectiveJudgment(
            judge_name=judge_agent.name,
            target_name="",
            similarity_score=avg_score,
            trace_id=",".join(trace_ids),
            raw_response=last_response,
            chunk_scores=chunk_scores,
            num_questions_shown=len(pairs),
        )

    # -- full matrix computation ---------------------------------------------

    def compute_subjective_matrix(
        self,
        agent_configs: dict[str, dict],
        benchmark_results: dict[str, BenchmarkResult],
    ) -> tuple[dict[tuple[str, str], float], list[SubjectiveJudgment]]:
        """Compute the full (potentially asymmetric) subjective similarity matrix.

        For every ordered pair (A, B) where A != B:
          1. Extract B's QAPairs from benchmark_results[B]
          2. Create a fresh instance of A (judging instance)
          3. Show A only B's responses
          4. A rates similarity 0-100

        Self-similarity is always 100.

        Args:
            agent_configs: Map of agent_name -> agent config dict.
            benchmark_results: Map of agent_name -> BenchmarkResult.

        Returns:
            Tuple of (sim_matrix, all_judgments).
            sim_matrix maps (judge_name, target_name) -> similarity score.
        """
        from src.registry.agent_registry import create_agent

        agent_names = list(agent_configs.keys())

        # Extract QAPairs for each agent
        qa_by_agent: dict[str, list[QAPair]] = {}
        for name in agent_names:
            result = benchmark_results[name]
            qa_by_agent[name] = LLMJudge.extract_qa_pairs(result.raw_responses)

        # Build ordered task list: (judge_name, target_name)
        # Skip RandomAgent as judge — it can't meaningfully rate similarity
        tasks = [
            (judge_name, target_name)
            for judge_name in agent_names
            for target_name in agent_names
            if judge_name != target_name
            and "RandomAgent" not in agent_configs[judge_name].get("type", "")
        ]

        def _run_judgment(
            task: tuple[str, str],
        ) -> tuple[str, str, float, SubjectiveJudgment]:
            judge_name, target_name = task
            cfg = agent_configs[judge_name]
            judge = create_agent(
                {
                    **cfg,
                    "player_id": 99,
                    "log_file": "subjective_judge_log.txt",
                }
            )
            judgment = self._judge_single(judge, qa_by_agent[target_name])
            judgment.judge_name = judge_name
            judgment.target_name = target_name
            return judge_name, target_name, judgment.similarity_score, judgment

        results = run_tasks(
            tasks,
            _run_judgment,
            max_workers=self.max_workers,
            desc="subjective similarity judging",
        )

        sim_matrix: dict[tuple[str, str], float] = {}
        all_judgments: list[SubjectiveJudgment] = []

        for judge_name, target_name, score, judgment in results:
            sim_matrix[(judge_name, target_name)] = score
            all_judgments.append(judgment)

        # Self-similarity = 100
        for name in agent_names:
            sim_matrix[(name, name)] = 100.0

        return sim_matrix, all_judgments
