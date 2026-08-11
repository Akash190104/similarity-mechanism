"""Agent adapters backed by Inspect AI's Model abstraction.

These classes implement the existing ``Agent`` ABC from
``src.agents.agent_manager`` but route LLM calls through Inspect AI's
``Model.generate()`` instead of the custom ``LLMManager``/``ClientAPILLM``
stack.  This lets the entire existing game/mechanism/benchmark pipeline run
unchanged while benefiting from Inspect's model support, logging, and CLI.
"""

from __future__ import annotations

import asyncio
import copy
import textwrap
import uuid
from typing import Any

from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)

from src.agents.agent_manager import Agent, RandomAgent
from src.logger_manager import LOGGER


# ---------------------------------------------------------------------------
# Async-to-sync bridge
# ---------------------------------------------------------------------------

_thread_local = __import__("threading").local()


def _run_async(coro):
    """Run an async coroutine from synchronous code.

    Handles the common case where an event loop may or may not already be
    running (e.g. inside Inspect's own async runtime).  When called from
    plain threads (e.g. a ThreadPoolExecutor), each thread reuses a
    persistent event loop to avoid the overhead of creating/destroying
    one per call.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an existing event loop (e.g. Inspect's task runner).
        # Create a new thread to run the coroutine to avoid deadlock.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        # Reuse a per-thread event loop for efficiency when called from
        # a ThreadPoolExecutor with many workers.
        tl_loop = getattr(_thread_local, "loop", None)
        if tl_loop is None or tl_loop.is_closed():
            tl_loop = asyncio.new_event_loop()
            _thread_local.loop = tl_loop
        return tl_loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Base Inspect Agent
# ---------------------------------------------------------------------------

class InspectAgent(Agent):
    """Agent backed by an Inspect AI ``Model``.

    Bypasses the parent ``__init__`` (which uses ``LLMManager``) and instead
    stores an Inspect ``Model`` instance directly.  The rest of the ``Agent``
    interface (``chat_with_retries``, ``name``, ``serialize``, etc.) works
    unchanged.
    """

    def __init__(self, model: Model, agent_config: dict) -> None:
        # Do NOT call super().__init__ — it would try to instantiate an LLM
        # via LLMManager.  Instead, set every attribute the parent expects.
        self.model = model
        self.model_type: str = agent_config["llm"]["model"]
        self.kwargs: dict = agent_config["llm"].get("kwargs", {})
        self.player_id: int = agent_config["player_id"]
        self.agent_config: dict = agent_config
        self.log_file: str = agent_config.get("log_file", "game_log.txt")

    # -- Core LLM call -----------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Call Inspect's Model.generate() synchronously and return the text."""
        config = GenerateConfig(**{
            k: v for k, v in self.kwargs.items()
            if k in ("temperature", "max_tokens", "top_p", "top_k",
                      "stop_seqs", "frequency_penalty", "presence_penalty",
                      "seed", "num_choices", "reasoning_effort",
                      "reasoning_tokens", "effort")
        })
        output = _run_async(
            self.model.generate(
                input=[ChatMessageUser(content=prompt)],
                config=config,
            )
        )
        return output.completion

    def invoke(self, messages: str) -> tuple[str, str]:
        """Invoke the model with logging.  Returns ``(response, trace_id)``."""
        trace_id = str(uuid.uuid4())[:8]
        response = self._generate(messages)
        self._log_inference(messages, response, trace_id)
        return response, trace_id

    def chat(self, messages: str) -> tuple[str, str]:
        """Subclasses override this to add prompt wrapping."""
        return self.invoke(messages)

    @property
    def agent_type(self) -> str:
        return f"{self.model_type}(Inspect)"


# ---------------------------------------------------------------------------
# CoT / IO variants (mirror existing prompt strategies)
# ---------------------------------------------------------------------------

class InspectIOAgent(InspectAgent):
    """Inspect-backed IOAgent — appends a concise output-only instruction."""

    def chat(self, messages: str) -> tuple[str, str]:
        messages += textwrap.dedent(
            """
            Please ONLY provide the output to the above question.
            DO NOT provide any additional text or explanation.
            """
        )
        return self.invoke(messages)

    @property
    def agent_type(self) -> str:
        return f"{self.model_type}(IO)"


class InspectCoTAgent(InspectAgent):
    """Inspect-backed CoTAgent — appends chain-of-thought instructions."""

    def chat(self, messages: str) -> tuple[str, str]:
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
        return f"{self.model_type}(CoT)"


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

INSPECT_AGENT_REGISTRY: dict[str, type[InspectAgent]] = {
    "IOAgent": InspectIOAgent,
    "CoTAgent": InspectCoTAgent,
}


def _model_name_for_inspect(agent_cfg: dict) -> str:
    """Convert an agent config's provider/model into an Inspect model string.

    Inspect uses ``"provider/model"`` format.  The existing configs already
    store the model in that format for OpenRouter (e.g.
    ``google/gemini-3.1-flash-lite-preview``).  This function prepends the
    Inspect provider prefix when needed.
    """
    llm = agent_cfg["llm"]
    provider = llm["provider"]
    model = llm["model"]

    # Map our provider names to Inspect provider prefixes
    provider_map = {
        "OpenAI": "openai",
        "Gemini": "google",
        "OpenRouter": "openrouter",
        "HFInstance": "hf",
    }

    prefix = provider_map.get(provider)
    if prefix is None:
        raise ValueError(
            f"Unsupported provider {provider!r} for Inspect adapter. "
            f"Supported: {list(provider_map)}"
        )

    # Avoid double-prefixing (e.g. OpenRouter models already have org/model)
    if provider == "OpenRouter":
        return f"openrouter/{model}"
    if provider == "OpenAI" and "/" not in model:
        return f"openai/{model}"
    if provider == "Gemini" and not model.startswith("google/"):
        return f"google/{model}"
    if provider == "HFInstance":
        return f"hf/{model}"

    return f"{prefix}/{model}"


def create_inspect_agent(agent_config: dict) -> Agent:
    """Create a single Inspect-backed agent from a config dict.

    For ``RandomAgent`` and ``TestInstance`` configs, falls back to the
    original agent factory (no Inspect model needed).  For all other types,
    creates an ``InspectCoTAgent`` or ``InspectIOAgent`` backed by Inspect's
    model layer.
    """
    agent_type = agent_config["type"]

    if agent_type == "RandomAgent":
        return RandomAgent(agent_config)

    # TestInstance: use the original agent factory (fake LLM, no API calls)
    provider = agent_config.get("llm", {}).get("provider", "")
    if provider == "TestInstance":
        from src.registry.agent_registry import create_agent
        return create_agent(agent_config)

    agent_class = INSPECT_AGENT_REGISTRY.get(agent_type)
    if agent_class is None:
        raise ValueError(
            f"Unknown agent type {agent_type!r}. "
            f"Supported: {list(INSPECT_AGENT_REGISTRY)} + RandomAgent"
        )

    model_str = _model_name_for_inspect(agent_config)
    config = GenerateConfig(**{
        k: v for k, v in agent_config["llm"].get("kwargs", {}).items()
        if k in ("temperature", "max_tokens", "top_p", "top_k",
                  "stop_seqs", "frequency_penalty", "presence_penalty",
                  "seed", "num_choices", "reasoning_effort",
                  "reasoning_tokens", "effort")
    })
    model = get_model(model_str, config=config, memoize=False)
    return agent_class(model, agent_config)


def create_inspect_agents(
    agent_cfgs: list[dict], num_players: int
) -> list[Agent]:
    """Create Inspect-backed players from agent configurations.

    Mirrors ``create_players_with_player_id`` from
    ``src.registry.agent_registry`` but uses Inspect models.
    """
    players: list[Agent] = []
    for cfg in agent_cfgs:
        if "player_id" in cfg:
            agent = create_inspect_agent(copy.deepcopy(cfg))
            players.append(agent)
        else:
            for player_id in range(1, num_players + 1):
                agent_config = copy.deepcopy(cfg)
                agent_config["player_id"] = player_id
                agent = create_inspect_agent(agent_config)
                players.append(agent)
    return players
