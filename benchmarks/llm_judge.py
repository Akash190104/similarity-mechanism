"""LLM-as-judge similarity scoring.

Given benchmark results from two agents, asks an LLM to judge how similar
their decision-making is. Benchmark-agnostic — works with any benchmark
that stores question_text and chosen_answer_text in raw_responses.
"""

import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks.base import BenchmarkResult
from src.utils.concurrency import run_tasks

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class QAPair:
    """A single question-answer pair from a benchmark."""

    question_text: str
    chosen_answer_text: str
    answer_options: list[str] | None = None  # e.g. ["A) ...", "B) ...", ...]
    full_response: str | None = None


@dataclass
class JudgmentResult:
    """Result from one model judging the similarity of a pair."""

    judge_model: str
    similarity_score: float  # 0-100 (average if chunked)
    llm1_is: str  # which real model was labeled "LLM1"
    trace_id: str  # comma-separated if chunked
    raw_response: str  # last chunk's response (or only response)
    chunk_scores: list[float] | None = None  # per-chunk scores when chunked


@dataclass
class PairJudgment:
    """Both models' judgments for one pair."""

    judgment_by_a: JudgmentResult
    judgment_by_b: JudgmentResult


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LLM_JUDGE_PROMPT_ANSWERS = """\
You are evaluating how similarly two language models think and reason. \
Below are their responses to the same set of questions. The models are \
labeled "LLM1" and "LLM2" — you do not know which model is which.

For each question, you can see what answer each model chose.

{qa_dossier}

Based on these responses, how similar are these two models in their \
decision-making style? Consider:
- Do they tend to reach the same conclusions?
- Do they seem to apply similar reasoning strategies?
- Do they show similar biases or preferences?

Provide a similarity score from 0 to 100, where:
- 0 means completely different reasoning styles
- 50 means moderately similar
- 100 means nearly identical reasoning styles

Think step by step about the patterns you observe, then provide your \
final score.

Your response MUST end with exactly: SIMILARITY SCORE: <number>"""

LLM_JUDGE_PROMPT_REASONING = """\
You are evaluating how similarly two language models think and reason. \
Below are their full responses to the same set of questions, including \
their step-by-step reasoning. The models are labeled "LLM1" and "LLM2" \
— you do not know which model is which.

For each question, you can see each model's full reasoning process and \
their final answer.

{qa_dossier}

Based on these responses, how similar are these two models in their \
reasoning and decision-making style? Consider:
- Do they follow similar chains of reasoning?
- Do they tend to reach the same conclusions?
- Do they weigh similar factors when making decisions?
- Do they show similar biases or preferences?

Provide a similarity score from 0 to 100, where:
- 0 means completely different reasoning styles
- 50 means moderately similar
- 100 means nearly identical reasoning styles

Think step by step about the patterns you observe, then provide your \
final score.

Your response MUST end with exactly: SIMILARITY SCORE: <number>"""


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Ask an LLM to judge similarity between two agents based on benchmark responses."""

    def __init__(
        self,
        mode: str = "answers_only",
        max_questions: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        """
        Args:
            mode: "answers_only" or "reasoning_trace".
            max_questions: Limit how many questions are shown to the judge.
            chunk_size: If set, split questions into chunks of this size,
                judge each chunk independently, and average the scores.
                Useful when the full dossier exceeds context window limits.
        """
        if mode not in ("answers_only", "reasoning_trace"):
            raise ValueError(f"Unknown judge mode: {mode}")
        self.mode = mode
        self.max_questions = max_questions
        self.chunk_size = chunk_size

    # -- extraction ----------------------------------------------------------

    @staticmethod
    def _extract_answer(r: dict) -> str:
        """Extract the answer text from a raw_response entry.

        Benchmarks use different field names:
        - chosen_answer_text: newcomb, dilemmas, moral_choice, hle, gpqa
        - chosen_letter: trait, multi_tp
        - rating: cabin, ggb
        - parsed_tosses / parsed_rolls: random coin/die
        - action_distribution: similarity_game
        """
        if "chosen_answer_text" in r:
            return str(r["chosen_answer_text"])
        if "chosen_letter" in r:
            return str(r["chosen_letter"])
        if "rating" in r:
            return str(r["rating"])
        if "parsed_tosses" in r:
            return ", ".join(str(t) for t in r["parsed_tosses"])
        if "parsed_rolls" in r:
            return ", ".join(str(d) for d in r["parsed_rolls"])
        if "action_distribution" in r:
            return str(r["action_distribution"])
        return str(r.get("full_response", ""))

    @staticmethod
    def extract_qa_pairs(raw_responses: list[dict]) -> list[QAPair]:
        """Build QAPair list from any benchmark's raw_responses.

        Handles different answer field names across benchmarks.
        Optionally reads ``full_response``.
        """
        pairs: list[QAPair] = []
        for r in raw_responses:
            pairs.append(
                QAPair(
                    question_text=r.get("question_text", r.get("prompt", "")),
                    chosen_answer_text=LLMJudge._extract_answer(r),
                    answer_options=r.get("answer_options"),
                    full_response=r.get("full_response"),
                )
            )
        return pairs

    # -- prompt building -----------------------------------------------------

    def _build_judge_prompt(
        self,
        qa_a: list[QAPair],
        qa_b: list[QAPair],
    ) -> tuple[str, str]:
        """Build the comparison prompt with random LLM1/LLM2 assignment.

        Returns (prompt, llm1_is) where llm1_is is "a" or "b".
        """
        # Random assignment to prevent position bias
        llm1_is = random.choice(["a", "b"])
        if llm1_is == "a":
            qa_1, qa_2 = qa_a, qa_b
        else:
            qa_1, qa_2 = qa_b, qa_a

        # Limit questions if configured
        pairs = list(zip(qa_1, qa_2))
        if self.max_questions and len(pairs) > self.max_questions:
            pairs = random.sample(pairs, self.max_questions)

        dossier = self._format_dossier(pairs)

        if self.mode == "reasoning_trace":
            prompt = LLM_JUDGE_PROMPT_REASONING.format(qa_dossier=dossier)
        else:
            prompt = LLM_JUDGE_PROMPT_ANSWERS.format(qa_dossier=dossier)

        return prompt, llm1_is

    @staticmethod
    def _strip_final_answer(text: str) -> str:
        """Remove final answer indicators from reasoning traces.

        Strips patterns like "FINAL ANSWER: B.", "My answer is A",
        "I choose B", "I select C." etc. so the judge only sees reasoning.
        """
        # Remove "FINAL ANSWER: X." and everything after
        text = re.sub(
            r"FINAL ANSWER:\s*[A-Z]\.?.*", "", text, flags=re.IGNORECASE | re.DOTALL
        )
        # Remove common answer declaration patterns at the end
        text = re.sub(
            r"(?:my (?:final )?answer is|I (?:choose|select|pick|go with))\s*[A-Z]\.?.*$",
            "", text, flags=re.IGNORECASE | re.MULTILINE,
        )
        return text.rstrip()

    def _format_dossier(self, pairs: list[tuple[QAPair, QAPair]]) -> str:
        """Format Q&A pairs into a dossier string."""
        lines: list[str] = []
        for i, (q1, q2) in enumerate(pairs, 1):
            lines.append(f"--- Question {i} ---")
            lines.append(f"Q: {q1.question_text}")

            # Show answer options if available (use q1's options — same question)
            if q1.answer_options:
                lines.append("Options:")
                for opt in q1.answer_options:
                    lines.append(f"  {opt}")

            if self.mode == "reasoning_trace" and q1.full_response and q2.full_response:
                reasoning_1 = self._strip_final_answer(q1.full_response)
                reasoning_2 = self._strip_final_answer(q2.full_response)
                lines.append(f"\nLLM1's reasoning:\n{reasoning_1}")
                lines.append(f"\nLLM2's reasoning:\n{reasoning_2}")
            else:
                lines.append(f"LLM1's answer: {q1.chosen_answer_text}")
                lines.append(f"LLM2's answer: {q2.chosen_answer_text}")

            lines.append("")
        return "\n".join(lines)

    # -- parsing -------------------------------------------------------------

    @staticmethod
    def _parse_score(response: str) -> int:
        """Parse similarity score from judge response.

        Raises ValueError if no valid score found (triggers retry).
        """
        # Primary: SIMILARITY SCORE: <number>
        match = re.search(r"SIMILARITY SCORE:\s*(\d+)", response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score

        # Fallback: last integer in 0-100 range
        all_ints = re.findall(r"\b(\d{1,3})\b", response)
        for candidate in reversed(all_ints):
            val = int(candidate)
            if 0 <= val <= 100:
                return val

        raise ValueError(
            "Could not parse a similarity score (0-100) from the response. "
            "Please end your response with: SIMILARITY SCORE: <number>"
        )

    # -- judging -------------------------------------------------------------

    def judge_pair(
        self,
        judge_agent: Any,
        qa_a: list[QAPair],
        qa_b: list[QAPair],
    ) -> JudgmentResult:
        """Have one agent judge the similarity between two sets of responses.

        If chunk_size is set, splits questions into chunks, judges each
        independently, and averages the scores.
        """
        if self.chunk_size and len(qa_a) > self.chunk_size:
            return self._judge_pair_chunked(judge_agent, qa_a, qa_b)

        prompt, llm1_is = self._build_judge_prompt(qa_a, qa_b)
        response, trace_id, score = judge_agent.chat_with_retries(
            prompt, self._parse_score
        )
        return JudgmentResult(
            judge_model=judge_agent.name,
            similarity_score=float(score),
            llm1_is=llm1_is,
            trace_id=trace_id,
            raw_response=response,
        )

    def _judge_pair_chunked(
        self,
        judge_agent: Any,
        qa_a: list[QAPair],
        qa_b: list[QAPair],
    ) -> JudgmentResult:
        """Judge in chunks and average scores."""
        assert self.chunk_size is not None
        # Apply max_questions first, then chunk
        pairs = list(zip(qa_a, qa_b))
        if self.max_questions and len(pairs) > self.max_questions:
            pairs = random.sample(pairs, self.max_questions)

        # Split into chunks
        chunks = [
            pairs[i : i + self.chunk_size]
            for i in range(0, len(pairs), self.chunk_size)
        ]

        # Use consistent LLM1/LLM2 assignment across all chunks
        llm1_is = random.choice(["a", "b"])

        chunk_scores: list[float] = []
        trace_ids: list[str] = []
        last_response = ""

        from tqdm import tqdm
        for chunk_idx, chunk in enumerate(tqdm(
            chunks, desc=f"  chunks [{judge_agent.name.split('/')[-1][:15]}]", leave=False
        )):
            if llm1_is == "a":
                chunk_qa_1 = [p[0] for p in chunk]
                chunk_qa_2 = [p[1] for p in chunk]
            else:
                chunk_qa_1 = [p[1] for p in chunk]
                chunk_qa_2 = [p[0] for p in chunk]

            dossier = self._format_dossier(list(zip(chunk_qa_1, chunk_qa_2)))

            if self.mode == "reasoning_trace":
                prompt = LLM_JUDGE_PROMPT_REASONING.format(qa_dossier=dossier)
            else:
                prompt = LLM_JUDGE_PROMPT_ANSWERS.format(qa_dossier=dossier)

            # Add chunk context
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
        return JudgmentResult(
            judge_model=judge_agent.name,
            similarity_score=avg_score,
            llm1_is=llm1_is,
            trace_id=",".join(trace_ids),
            raw_response=last_response,
            chunk_scores=chunk_scores,
        )

    def judge_all_pairs(
        self,
        agents: list[Any],
        agent_configs: list[dict],
        benchmark_results: dict[str, BenchmarkResult],
        max_workers: int | None = None,
    ) -> dict[tuple[str, str], PairJudgment]:
        """Judge similarity for all unordered pairs. Each model judges independently.

        Args:
            agents: List of agents (used for name lookup).
            agent_configs: Corresponding config dicts for creating judge instances.
            benchmark_results: Map of agent name to BenchmarkResult.
            max_workers: Concurrency for judge calls.

        Returns:
            Dict mapping (agent_a_name, agent_b_name) to PairJudgment.
        """
        from itertools import combinations

        from src.registry.agent_registry import create_agent

        # Extract Q&A pairs for each agent
        qa_by_agent: dict[str, list[QAPair]] = {}
        for agent in agents:
            result = benchmark_results[agent.name]
            qa_by_agent[agent.name] = self.extract_qa_pairs(result.raw_responses)

        # Build all judge tasks: (agent_config, agent_name, qa_a, qa_b, pair_key)
        tasks: list[tuple] = []
        for (a, cfg_a), (b, cfg_b) in combinations(
            zip(agents, agent_configs), 2
        ):
            # A judges the pair
            tasks.append((cfg_a, a.name, qa_by_agent[a.name], qa_by_agent[b.name], (a.name, b.name), "a"))
            # B judges the pair
            tasks.append((cfg_b, b.name, qa_by_agent[a.name], qa_by_agent[b.name], (a.name, b.name), "b"))

        def _run_judgment(task: tuple) -> tuple:
            cfg, _name, qa_a, qa_b, pair_key, role = task
            judge = create_agent({**cfg, "player_id": 2, "log_file": "judge_log.txt"})
            result = self.judge_pair(judge, qa_a, qa_b)
            return pair_key, role, result

        results = run_tasks(
            tasks,
            _run_judgment,
            max_workers=max_workers,
            desc="judging similarity",
        )

        # Assemble PairJudgments
        partial: dict[tuple[str, str], dict[str, JudgmentResult]] = {}
        for pair_key, role, result in results:
            partial.setdefault(pair_key, {})[role] = result

        judgments: dict[tuple[str, str], PairJudgment] = {}
        for pair_key, roles in partial.items():
            judgments[pair_key] = PairJudgment(
                judgment_by_a=roles["a"],
                judgment_by_b=roles["b"],
            )

        return judgments
