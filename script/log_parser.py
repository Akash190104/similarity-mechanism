"""Parse game_log.txt files into structured entries for filtering and display."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LogEntry:
    """A single prompt or response entry from a game log."""

    trace_id: str
    timestamp: str
    agent: str
    entry_type: str  # "prompt" or "response"
    content: str
    round_number: int | None = None
    similarity_pct: int | None = None


@dataclass
class LogPair:
    """A matched prompt-response pair."""

    trace_id: str
    timestamp: str
    agent: str
    prompt: str
    response: str
    round_number: int | None = None
    similarity_pct: int | None = None
    model: str = ""  # extracted from agent name, e.g., "openai/gpt-4o"


# Regex for parsing log headers
_HEADER_RE = re.compile(
    r"=====\s+(Prompt|Response)\s+\[ID:\s+([a-f0-9]+)\]\s+\[([^\]]+)\]\s+====="
)
_AGENT_RE = re.compile(r"^agent:\s+(.+)$", re.MULTILINE)
_ROUND_RE = re.compile(r"played this game for (\d+) round\(s\)")
_SIMILARITY_RE = re.compile(r"opponent is (\d+)% similar")


def parse_game_log(path: Path) -> list[LogEntry]:
    """Parse a game_log.txt into structured LogEntry objects.

    Args:
        path: Path to the game_log.txt file.

    Returns:
        List of LogEntry objects, one per prompt/response block.
    """
    text = path.read_text(encoding="utf-8")
    entries: list[LogEntry] = []

    # Split on headers
    parts = _HEADER_RE.split(text)
    # parts layout: [preamble, type1, id1, ts1, body1, type2, id2, ts2, body2, ...]

    i = 1  # skip preamble
    while i + 3 < len(parts):
        entry_type = parts[i].lower()  # "prompt" or "response"
        trace_id = parts[i + 1]
        timestamp = parts[i + 2]
        body = parts[i + 3]
        i += 4

        # Extract agent name
        agent_match = _AGENT_RE.search(body)
        agent = agent_match.group(1).strip() if agent_match else ""

        # Extract content (everything after the agent/type line)
        if entry_type == "prompt":
            content_match = re.search(r"^prompt:\s*\n", body, re.MULTILINE)
        else:
            content_match = re.search(r"^response:\s*\n", body, re.MULTILINE)

        if content_match:
            content = body[content_match.end():].strip()
        else:
            content = body.strip()

        # Try to extract round number from prompt content
        round_number = None
        round_match = _ROUND_RE.search(content)
        if round_match:
            round_number = int(round_match.group(1))

        # Try to extract similarity percentage
        similarity_pct = None
        sim_match = _SIMILARITY_RE.search(content)
        if sim_match:
            similarity_pct = int(sim_match.group(1))

        entries.append(
            LogEntry(
                trace_id=trace_id,
                timestamp=timestamp,
                agent=agent,
                entry_type=entry_type,
                content=content,
                round_number=round_number,
                similarity_pct=similarity_pct,
            )
        )

    return entries


def pair_entries(entries: list[LogEntry]) -> list[LogPair]:
    """Match prompt-response entries by trace_id into LogPair objects.

    Args:
        entries: List of LogEntry objects from parse_game_log.

    Returns:
        List of LogPair objects with matched prompt and response.
    """
    prompts: dict[str, LogEntry] = {}
    pairs: list[LogPair] = []

    for entry in entries:
        if entry.entry_type == "prompt":
            prompts[entry.trace_id] = entry
        elif entry.entry_type == "response":
            prompt_entry = prompts.get(entry.trace_id)
            if prompt_entry:
                # Extract model name from agent (e.g., "openai/gpt-4o(CoT)#P1" -> "openai/gpt-4o")
                model = re.sub(r"\(.*$", "", entry.agent)

                pairs.append(
                    LogPair(
                        trace_id=entry.trace_id,
                        timestamp=entry.timestamp,
                        agent=entry.agent,
                        prompt=prompt_entry.content,
                        response=entry.content,
                        round_number=prompt_entry.round_number,
                        similarity_pct=prompt_entry.similarity_pct,
                        model=model,
                    )
                )

    return pairs


def extract_metadata(pairs: list[LogPair]) -> dict:
    """Extract unique filter values from parsed log pairs.

    Args:
        pairs: List of LogPair objects.

    Returns:
        Dict with keys: models, agents, round_numbers, has_similarity,
        similarity_values, min_round, max_round.
    """
    models = sorted(set(p.model for p in pairs if p.model))
    agents = sorted(set(p.agent for p in pairs if p.agent))
    rounds = sorted(set(p.round_number for p in pairs if p.round_number is not None))
    sim_values = sorted(set(p.similarity_pct for p in pairs if p.similarity_pct is not None))

    return {
        "models": models,
        "agents": agents,
        "round_numbers": rounds,
        "has_similarity": bool(sim_values),
        "similarity_values": sim_values,
        "min_round": min(rounds) if rounds else 0,
        "max_round": max(rounds) if rounds else 0,
    }
