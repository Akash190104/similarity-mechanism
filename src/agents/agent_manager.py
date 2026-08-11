"""Agent abstractions and shared LLM caching utilities."""
import itertools
import textwrap
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

from tqdm import tqdm

from src.agents.base import LLM
from src.agents.client_api_llm import ClientAPILLM
from src.agents.hf_llm import HFInstance
from src.agents.test_llm import TestInstance
from src.logger_manager import LOGGER


class LLMManager:
    """Central cache for LLM backends, keyed by (model, provider)."""

    def __init__(self) -> None:
        self.llms = dict()

    def get_llm(self, model_name: str, provider: str) -> LLM:
        """Return a cached ``LLM`` implementation for ``model_name``/``provider``."""
        if model_name not in self.llms:
            if provider == "HFInstance":
                self.llms[model_name] = HFInstance(model_name)
            elif provider == "TestInstance":
                self.llms[model_name] = TestInstance()
            elif provider in {"OpenAI", "Gemini", "OpenRouter"}:
                # Default to OpenAI API based models
                self.llms[model_name] = ClientAPILLM(model_name, provider)
            else:
                raise ValueError(
                    f"Unknown provider {provider} for model {model_name}"
                )
        return self.llms[model_name]


class Agent(ABC):
    """
    Abstract base class for an LLM-based agent.
    """

    llm_manager = LLMManager()
    _instance_counter = itertools.count(1)

    def __init__(self, agent_config: dict) -> None:
        llm_config = agent_config["llm"]
        self.model_type = llm_config["model"]
        self.kwargs = llm_config.get("kwargs", {})
        self.pipeline = type(self).llm_manager.get_llm(
            self.model_type, llm_config["provider"]
        )
        self.player_id: int = agent_config["player_id"]
        self.agent_config = agent_config
        self.log_file: str = agent_config.get("log_file", "game_log.txt")

    @abstractmethod
    def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Chat with the agent using the provided messages."""
        raise NotImplementedError

    def invoke(self, messages: str) -> tuple[str, str]:
        """Invoke the agent's LLM pipeline with logging. Returns response and trace_id."""
        trace_id = str(uuid.uuid4())[:8]

        response = self.pipeline.invoke(messages, **self.kwargs)

        self._log_inference(messages, response, trace_id)
        return response, trace_id

    def _format_fix_invoke(self, messages: str) -> tuple[str, str]:
        """Ultra-cheap call used ONLY by the format-fix retry path.

        Bypasses both the CoT/IO wrapper (no 'think step by step' suffix) and
        any reasoning budget the model normally uses. The format-fix prompt
        already contains the model's prior reasoning + the parse error, so
        the model just needs to emit the formatted final answer — no further
        deliberation required. This is what the user explicitly asked for:
        re-use the existing reasoning, ask only for a corrected output.
        """
        trace_id = str(uuid.uuid4())[:8]
        cheap_kwargs = {k: v for k, v in self.kwargs.items() if k != "reasoning"}
        response = self.pipeline.invoke(messages, **cheap_kwargs)
        self._log_inference(messages, response, trace_id)
        return response, trace_id

    def _log_inference(self, prompt: str, response: str, trace_id: str) -> None:
        """Log the inference to the game log."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = (
            f"===== Prompt [ID: {trace_id}] [{timestamp}] =====\n"
            f"agent: {self.name}\n"
            "prompt:\n"
            f"{prompt}\n"
            f"===== Response [ID: {trace_id}] [{timestamp}] =====\n"
            f"agent: {self.name}\n"
            "response:\n"
            f"{response}\n\n"
        )
        LOGGER.append_to_txt(entry, self.log_file)

    def chat_with_retries(
        self,
        base_prompt: str,
        parse_func: Callable[[str], Any],
        *,
        max_retries: int = 3,
    ) -> tuple[str, str, Any]:
        """Chat with the agent, retrying if response parsing fails.

        Retry ladder (cheaper → more expensive):
          1. *attempt 0*: original prompt.
          2. *attempt 1*: **format-fix** — shows the model its prior response and
             asks it to re-output ONLY the required final-answer block. Skips
             re-doing reasoning, which is a big speed-up for CoT agents whose
             reasoning was sound but whose final block was malformed.
          3. *attempts 2…max_retries*: traditional full-restate retry.
        """
        response = ""
        error_reason = ""

        for attempt in range(max_retries + 1):
            if attempt == 0:
                prompt = base_prompt
                response, trace_id = self.chat(prompt)
            elif attempt == 1 and max_retries >= 1:
                # Cheap format-fix: bypass CoT wrapper AND reasoning, since the
                # model's prior response already contains the reasoning we just
                # need formatted correctly.
                prompt = self._build_format_fix_prompt(
                    base_prompt, response, error_reason
                )
                response, trace_id = self._format_fix_invoke(prompt)
            else:
                prompt = self._build_retry_prompt(
                    base_prompt, response, error_reason
                )
                response, trace_id = self.chat(prompt)
            if not response or not response.strip():
                # Empty model output (e.g. reasoning consumed the whole token
                # budget). Re-calling just repeats the same deterministic
                # failure, so stop retrying and fail fast.
                error_reason = "empty response"
                break
            try:
                return response, trace_id, parse_func(response)
            except ValueError as e:
                error_reason = str(e)
                tqdm.write(
                    f"Attempt {attempt + 1} of {self.name} to parse response failed: "
                    f"{self._truncate_string(error_reason)} from response {self._truncate_string(response)!r}"
                )
        raise ValueError(
            f"Failed to parse response for {self.name} after {1 + max_retries} attempts. "
            f"Last error: {error_reason}. Last response: {response!r}"
        )

    @staticmethod
    def _truncate_string(s: str, max_chars: int = 300) -> str:
        """Truncate string to show first and last max_chars characters."""
        if len(s) <= 2 * max_chars:
            return s
        return f"{s[:max_chars]}...[truncated due to length]...{s[-max_chars:]}"

    @staticmethod
    def _build_retry_prompt(
        base_prompt: str, bad_response: str, error_reason: str
    ) -> str:
        """Restate the prompt, show prior response and ask for regeneration."""
        return (
            f"{base_prompt}\n\n"
            f"Your previous response was:\n{bad_response}\n\n"
            f"That response is INVALID because: {error_reason}\n\n"
            f"Please give the new output again!"
        )

    @staticmethod
    def _build_format_fix_prompt(
        base_prompt: str, bad_response: str, error_reason: str
    ) -> str:
        """Cheap format-only retry: skip re-doing reasoning, just ask the
        model to re-emit the final-answer block in the required format from
        the same reasoning it already produced.

        Embeds only the *format requirements* section of the base prompt so
        the model has the spec without paying for the full prompt re-read.
        """
        # Pull the trailing "Format requirement:" / "Instruction:" / "Your response
        # should be in the following format:" section if present, otherwise fall
        # back to the last 600 chars of the base prompt.
        spec = ""
        for marker in ("Format requirement:", "Instruction:",
                       "Your response should be in the following format:"):
            idx = base_prompt.rfind(marker)
            if idx != -1:
                spec = base_prompt[idx:]
                break
        if not spec:
            spec = base_prompt[-600:]

        return (
            "You produced the following response, but its final-answer block "
            "could not be parsed:\n\n"
            f"--- YOUR PRIOR RESPONSE ---\n{bad_response}\n--- END ---\n\n"
            f"Parse error: {error_reason}\n\n"
            "Re-output ONLY the final-answer block, in the exact required "
            "format below. Do NOT redo the reasoning — your reasoning was "
            "fine; only the format failed. Output the formatted answer with "
            "no extra prose.\n\n"
            f"{spec}"
        )

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the model type and prompt strategy."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"{self.agent_type}#P{self.player_id}"

    def serialize(self) -> dict:
        """Return the LLM configuration dictionary for this agent."""
        return self.agent_config

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name


class RandomAgent(Agent):
    """An agent that responds uniformly at random for any game or benchmark.

    No LLM backend needed.  Config format:

        - type: RandomAgent

    Handles every prompt type used in the codebase:
      • Game actions          → random probability distribution over actions
      • FINAL ANSWER: MCQ     → random valid letter (newcomb, trait, …)
      • Answer: {letter} MCQ  → random valid letter (dilemmas, moral_choice,
                                cabin A-E, ggb A-G, trait A-D, …)
      • HLE free-text         → random int 0-100 as answer
      • HLE judge             → CORRECT / INCORRECT at random
      • Coin-toss benchmark   → random H/T sequence
      • Die-roll benchmark    → random 1-6 sequence
      • Similarity judge      → random score 30-80
    """

    import re as _re
    import random as _random

    def __init__(self, agent_config: dict) -> None:
        self.kwargs: dict = {}
        self.player_id: int = agent_config["player_id"]
        self.agent_config = agent_config
        self.log_file: str = agent_config.get("log_file", "game_log.txt")

    def invoke(self, messages: str) -> tuple[str, str]:
        import random
        import re

        trace_id = str(uuid.uuid4())[:8]
        response = self._respond(messages, random, re)
        self._log_inference(messages, response, trace_id)
        return response, trace_id

    def chat(self, messages: str) -> tuple[str, str]:
        return self.invoke(messages)

    @staticmethod
    def _respond(prompt: str, random, re) -> str:
        # ── Game action distribution ──────────────────────────────────────────
        # Detect the JSON template used in game prompts: {"A0": <Int>, "A1": <Int>, ...}
        if re.search(r'"A0"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)', prompt):
            action_keys = re.findall(r'^\s*-\s*A(\d+)', prompt, re.MULTILINE)
            n = max(int(k) for k in action_keys) + 1 if action_keys else 2
            weights = [random.random() for _ in range(n)]
            total = sum(weights) or 1.0
            raw = [w / total * 100 for w in weights]
            ints = [int(v) for v in raw]
            diff = 100 - sum(ints)
            if diff:
                ints[0] += diff
            return str({f"A{i}": v for i, v in enumerate(ints)}).replace("'", '"')

        # ── Similarity / LLM judge ────────────────────────────────────────────
        if "SIMILARITY SCORE:" in prompt:
            return f"SIMILARITY SCORE: {random.randint(30, 80)}"

        # ── HLE judge (CORRECT / INCORRECT) ──────────────────────────────────
        if "Reply with exactly one word: CORRECT or INCORRECT" in prompt:
            return random.choice(["CORRECT", "INCORRECT"])

        # ── HLE benchmark (Explanation: / Answer: free-text or MCQ) ──────────
        if "Explanation: {your explanation" in prompt and "Answer: {" in prompt:
            options = re.findall(r"^([A-Z])\)\s+", prompt, re.MULTILINE)
            if options:
                return f"Explanation: Random response.\nAnswer: {random.choice(options)}"
            return f"Explanation: Random response.\nAnswer: {random.randint(0, 100)}"

        # ── Coin-toss benchmark ───────────────────────────────────────────────
        if re.search(r"flip a fair coin", prompt, re.IGNORECASE):
            m = re.search(r"(\d+)\s+times", prompt)
            n = int(m.group(1)) if m else 10
            return ", ".join(random.choice(["H", "T"]) for _ in range(n))

        # ── Die-roll benchmark ────────────────────────────────────────────────
        if re.search(r"roll a fair.*die", prompt, re.IGNORECASE):
            m = re.search(r"(\d+)\s+times", prompt)
            n = int(m.group(1)) if m else 10
            return ", ".join(str(random.randint(1, 6)) for _ in range(n))

        # ── newcomb / trait / gpqa: "FINAL ANSWER: X." format ────────────────
        if "FINAL ANSWER:" in prompt:
            options = re.findall(r"^([A-Z])\)\s+", prompt, re.MULTILINE)
            if not options:
                options = ["A", "B"]
            return f"FINAL ANSWER: {random.choice(options)}."

        # ── dilemmas / moral_choice / cabin / ggb / trait:
        #    "Answer: {the letter}" format ────────────────────────────────────
        if "Answer: {the letter" in prompt or (
            "Answer: {" in prompt and "Explanation: {" in prompt
        ):
            options = re.findall(r"^([A-Z])\)\s+", prompt, re.MULTILINE)
            if not options:
                options = ["A", "B"]
            return f"Explanation: Random response.\nAnswer: {random.choice(options)}"

        # ── Fallback: random number 0-100 ─────────────────────────────────────
        return str(random.randint(0, 100))

    @property
    def agent_type(self) -> str:
        return "RandomAgent"


class IOAgent(Agent):
    """Input/Output Agent.
    This agent is designed to be the most basic llm agent. Given a message, answer it.
    """

    def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Chat with the agent using the provided messages."""
        messages += textwrap.dedent(
            """
            Please ONLY provide the output to the above question.
            DO NOT provide any additional text or explanation.
            """
        )
        return self.invoke(messages)

    @property
    def agent_type(self) -> str:
        """Return the model type and prompt strategy."""
        return f"{self.model_type}(IO)"


class CoTAgent(Agent):
    """Chain-of-Thought Agent.

    This agent wraps the prompt to ask the LLM to think step-by-step.
    """

    def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Chat with the agent using the provided messages."""
        messages += textwrap.dedent(
            """
            Think about the question step by step.
            Break it down into small steps.
            Explain your reasoning, and then provide the final answer.
            """
        )
        return self.invoke(messages)

    @property
    def agent_type(self) -> str:
        """Return the model type and prompt strategy."""
        return f"{self.model_type}(CoT)"
