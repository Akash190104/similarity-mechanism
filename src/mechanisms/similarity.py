"""Similarity mechanism that frames opponents with configurable similarity prompts.

Supports multiple similarity sources:
- "fixed" (default): All pairs get the same told similarity percentage.
- "sweep": Play the game at each similarity level (0%, 10%, ..., 100%).
- "benchmark": Run a real benchmark, compute per-pair similarity, then play.
- "benchmark_sweep": Spoofed benchmark similarity — tell agents they scored X% on
  a benchmark without actually running it.
- "subjective": Run a benchmark, have each agent subjectively rate other agents'
  responses, then play with those (potentially asymmetric) similarity scores.
"""

import copy
import itertools
from datetime import datetime
from typing import Any, Sequence, override

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.games.base import Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism
from src.mechanisms.prompts import (
    SIMILARITY_MECHANISM_PROMPT_REPEATED,
    SIMILARITY_MECHANISM_PROMPT_SINGLE,
    SIMILARITY_NO_HISTORY,
    SIMILARITY_OTHERPLAYER_LABEL,
    SIMILARITY_ROUND_LINE,
    SIMILARITY_SELF_LABEL,
)
from src.mechanisms.similarity_utils import build_similarity_framing
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.ranking_evaluations.payoffs_base import PayoffsBase
from src.utils.concurrency import run_tasks
from src.utils.match_scheduler_reputation import RandomMatcher

VALID_SOURCES = ("fixed", "sweep", "benchmark", "benchmark_sweep", "subjective")


class Similarity(Mechanism):
    """Similarity mechanism that tells agents about their opponent's similarity.

    The ``similarity_source`` parameter controls where similarity values come from:

    - ``"fixed"`` (default): Every matchup uses the same ``similarity_pct``.
    - ``"sweep"``: Play the game at each similarity level (determined by
      ``increment`` or ``percentages``), collecting payoffs across all levels.
    - ``"benchmark"``: Run a real benchmark on each agent, compute per-pair
      similarity from benchmark results, then play the game with those values.
    - ``"benchmark_sweep"``: Tell agents they scored X% on a benchmark (without
      running it), sweeping across similarity levels and benchmarks.
    - ``"subjective"``: Run a benchmark, have each agent subjectively rate other
      agents' responses (decision-based or explanation-based), then play the game
      with those potentially asymmetric similarity scores.

    All modes play the actual game and produce ``MatchupPayoffs``.

    Prompt modes control how similarity is communicated to agents:

    - ``"percentage_updated"`` (default): Detailed decision-making overlap framing
    - ``"percentage"``: Simple "X% similar" framing
    - ``"domain"``: Domain-specific similarity
    - ``"vague"``: Non-specific similarity
    - ``"custom"``: Free-form custom text
    """

    def __init__(
        self,
        base_game: Game,
        *,
        # Source selection
        similarity_source: str = "fixed",
        # Common params
        similarity_pct: int = 70,
        prompt_mode: str = "percentage_updated",
        domain: str = "",
        custom_prompt: str = "",
        num_rounds: int = 1,
        discount: float = 0.8,
        # Sweep params
        increment: int = 10,
        percentages: list[int] | None = None,
        trials_per_level: int = 1,
        # Benchmark params
        benchmark: str | None = None,
        benchmark_kwargs: dict | None = None,
        benchmarks: list[str] | None = None,
        # Subjective params
        subjective_mode: str = "decision",
        subjective_chunk_size: int | None = None,
        subjective_max_items: int | None = None,
        # Framing: False/"similar" (default), True/"different", or "dissimilar"
        difference_framing: bool | str = False,
        # Benchmark cache (default: data/benchmark_cache/)
        benchmark_cache_dir: str | None = "auto",
        # Tournament workers
        tournament_workers: int = 1,
        # Per-model concurrency override: maps model name (with or without
        # the ``(CoT)``/``(IO)`` suffix) to its max concurrent API calls.
        # Models not listed fall back to ``tournament_workers``.
        model_workers: dict[str, int] | None = None,
        # Pair selection
        self_play_only: bool = False,
    ) -> None:
        if similarity_source not in VALID_SOURCES:
            raise ValueError(
                f"Unknown similarity_source: {similarity_source!r}. "
                f"Must be one of {VALID_SOURCES}"
            )
        super().__init__(base_game, tournament_workers=tournament_workers)
        self.similarity_source = similarity_source
        self.similarity_pct = similarity_pct
        self.prompt_mode = prompt_mode
        self.domain = domain
        self.custom_prompt = custom_prompt
        self.num_rounds = num_rounds
        self.discount = discount
        # Sweep
        self.increment = increment
        self.percentages = percentages
        self.trials_per_level = trials_per_level
        # Benchmark
        self.benchmark = benchmark
        self.benchmark_kwargs = benchmark_kwargs or {}
        self.benchmarks = benchmarks
        # Subjective
        self.subjective_mode = subjective_mode
        self.subjective_chunk_size = subjective_chunk_size
        self.subjective_max_items = subjective_max_items
        # Framing
        self.difference_framing = difference_framing
        # Pair selection
        self.self_play_only = self_play_only
        # Per-model concurrency overrides
        self.model_workers = model_workers or {}
        # Benchmark cache — resolve "auto" to project-level default
        if benchmark_cache_dir == "auto":
            from pathlib import Path
            self.benchmark_cache_dir = str(
                Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_cache"
            )
        else:
            self.benchmark_cache_dir = benchmark_cache_dir

    # ── Benchmark cache helpers ──────────────────────────────────────────

    def _cache_key(self, agent_id: str) -> str:
        """Return the cache file path for an agent's benchmark results."""
        import re
        safe = re.sub(r'[^\w\-.]', '_', agent_id)
        return f"benchmark_cache_{self.benchmark}_{safe}.json"

    def _load_cached_result(self, agent_id: str):
        """Load cached BenchmarkResult for agent_id, or None."""
        if not self.benchmark_cache_dir:
            return None
        import json
        from pathlib import Path
        from benchmarks.base import BenchmarkResult
        p = Path(self.benchmark_cache_dir) / self._cache_key(agent_id)
        if p.exists():
            data = json.loads(p.read_text())
            print(f"  [Cache HIT] Loaded {agent_id} from {p}")
            return BenchmarkResult.from_dict(data)
        return None

    def _save_cached_result(self, agent_id: str, result) -> None:
        """Save a BenchmarkResult to the cache directory."""
        if not self.benchmark_cache_dir:
            return
        import json
        from pathlib import Path
        p = Path(self.benchmark_cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        out = p / self._cache_key(agent_id)
        out.write_text(json.dumps(result.serialize(), indent=2))
        print(f"  [Cache SAVE] Saved {agent_id} to {out}")

    # ── Pair selection helpers ───────────────────────────────────────────

    def _filter_pairs(self, all_pairs: list) -> list:
        """When ``self_play_only`` is set, keep only pairs whose seats share an agent_type."""
        if not self.self_play_only:
            return all_pairs
        return [pair for pair in all_pairs
                if len({p.agent_type for p in pair}) == 1]

    # ── Per-model concurrency ────────────────────────────────────────────

    @staticmethod
    def _strip_agent_suffix(agent_type: str) -> str:
        """Strip ``(CoT)``/``(IO)``/etc. suffix so model_workers keys can match."""
        idx = agent_type.rfind("(")
        return agent_type[:idx] if idx >= 0 else agent_type

    def _build_model_semaphores(self, players: Sequence[Agent]):
        """Build per-model ``Semaphore`` map for throttling, plus total pool cap.

        Returns ``(sems, pool_cap)`` where ``sems`` maps agent_type → Semaphore
        and ``pool_cap`` is the sum of per-model caps (the right top-level
        ThreadPoolExecutor size). Returns ``(None, tournament_workers)`` when
        ``model_workers`` is empty.
        """
        if not self.model_workers:
            return None, self.tournament_workers
        from threading import Semaphore
        sems: dict[str, "Semaphore"] = {}
        total = 0
        for p in players:
            at = p.agent_type
            if at in sems:
                continue
            cap = (
                self.model_workers.get(at)
                or self.model_workers.get(self._strip_agent_suffix(at))
                or self.tournament_workers
            )
            sems[at] = Semaphore(cap)
            total += cap
        return sems, total

    @staticmethod
    def _acquire_for_pair(sems, pair):
        """Acquire all (deduplicated) per-agent semaphores for one task. Returns the list released later."""
        if sems is None:
            return []
        unique = list({p.agent_type for p in pair})
        for at in unique:
            sems[at].acquire()
        return unique

    # ── Framing helpers ───────────────────────────────────────────────────

    def _build_similarity_framing(self, similarity_pct: int | None = None) -> str:
        """Build framing string. Uses self.similarity_pct if not overridden."""
        pct = similarity_pct if similarity_pct is not None else self.similarity_pct
        return build_similarity_framing(
            similarity_pct=pct,
            prompt_mode=self.prompt_mode,
            domain=self.domain,
            custom_prompt=self.custom_prompt,
            num_other_players=self.base_game.num_players - 1,
            difference_framing=self.difference_framing,
        )

    def _build_per_player_framing(
        self, player: Agent, players: Sequence[Agent], similarity_pct: int | None = None,
    ) -> str:
        """Build similarity framing for a specific player listing each other player."""
        from benchmarks.registry import resolve_framing_words
        from src.mechanisms.similarity_utils import _display_pct

        raw_pct = similarity_pct if similarity_pct is not None else self.similarity_pct
        pct = _display_pct(raw_pct, self.difference_framing)
        measure_word, relation_word = resolve_framing_words(self.difference_framing)
        others = [p for p in players if p is not player]
        lines = [
            f"- Player {other.player_id}'s decision-making is {pct}% {measure_word} yours."
            for other in others
        ]
        return (
            f"The following describes how {relation_word} each other player's"
            " decision-making is to yours:\n"
            + "\n".join(lines)
            + f"\nThis means how {relation_word} you and each other player reason and"
            " come to conclusions when facing the same strategic problem."
            " Note, however, all players are independently trying to"
            " maximize their own total points."
            " Remember, the other players are seeing this information as well."
        )

    def _build_prompts_single(
        self, players: Sequence[Agent], similarity_pct: int | None = None
    ) -> list[str]:
        """Build per-player prompts for single-shot mode."""
        if self.base_game.num_players > 2:
            return [
                SIMILARITY_MECHANISM_PROMPT_SINGLE.format(
                    similarity_framing=self._build_per_player_framing(
                        player, players, similarity_pct
                    ),
                )
                for player in players
            ]
        framing = self._build_similarity_framing(similarity_pct)
        prompt = SIMILARITY_MECHANISM_PROMPT_SINGLE.format(
            similarity_framing=framing,
        )
        return [prompt] * len(players)

    def _build_prompts_repeated(
        self,
        players: Sequence[Agent],
        round_idx: int,
        history: list[list[Move]],
        similarity_pct: int | None = None,
    ) -> list[str]:
        """Build per-player prompts for repeated mode with simple history."""
        rounds_played = round_idx - 1
        is_multiplayer = self.base_game.num_players > 2

        if not history:
            history_context = SIMILARITY_NO_HISTORY
            if is_multiplayer:
                return [
                    SIMILARITY_MECHANISM_PROMPT_REPEATED.format(
                        similarity_framing=self._build_per_player_framing(
                            player, players, similarity_pct
                        ),
                        discount=int(self.discount * 100),
                        round_idx=rounds_played,
                        history_context=history_context,
                    )
                    for player in players
                ]
            framing = self._build_similarity_framing(similarity_pct)
            prompt = SIMILARITY_MECHANISM_PROMPT_REPEATED.format(
                similarity_framing=framing,
                discount=int(self.discount * 100),
                round_idx=rounds_played,
                history_context=history_context,
            )
            return [prompt] * len(players)

        # Build per-player history views
        prompts = []
        for focus in players:
            history_lines = []
            for idx, round_moves in enumerate(history):
                focus_move = next(
                    (m for m in round_moves if m.player == focus), None
                )
                if focus_move is None:
                    continue

                actions = [
                    SIMILARITY_SELF_LABEL.format(
                        action_token=focus_move.action.to_token()
                    )
                ]
                for move in round_moves:
                    if move.player == focus:
                        continue
                    actions.append(
                        SIMILARITY_OTHERPLAYER_LABEL.format(
                            other_player=f"Player {move.player.player_id}",
                            action_token=move.action.to_token(),
                        )
                    )

                history_lines.append(
                    SIMILARITY_ROUND_LINE.format(
                        round_idx=idx + 1,
                        actions="\n".join(actions),
                    )
                )

            history_context = (
                "\n".join(reversed(history_lines))
                if history_lines
                else SIMILARITY_NO_HISTORY
            )

            if is_multiplayer:
                framing = self._build_per_player_framing(
                    focus, players, similarity_pct
                )
            else:
                framing = self._build_similarity_framing(similarity_pct)

            prompt = SIMILARITY_MECHANISM_PROMPT_REPEATED.format(
                similarity_framing=framing,
                discount=int(self.discount * 100),
                round_idx=rounds_played,
                history_context=history_context,
            )
            prompts.append(prompt)

        return prompts

    # ── Payoffs ────────────────────────────────────────────────────────────

    @override
    def _build_payoffs(self) -> PayoffsBase:
        if self.num_rounds > 1:
            return MatchupPayoffs(discount=self.discount)
        return MatchupPayoffs()

    # ── Core matchup play (used by all modes) ─────────────────────────────

    def _play_matchup_at(
        self,
        players: Sequence[Agent],
        similarity_pct: int,
    ) -> list[list[Move]]:
        """Play a matchup at a specific similarity percentage.

        Single-shot: one round, returns [[moves]].
        Repeated: multiple rounds with history, returns [[moves], ...].
        """
        if self.num_rounds == 1:
            prompts = self._build_prompts_single(players, similarity_pct)
            moves = self.base_game.play(
                additional_info=prompts, players=players
            )
            LOGGER.log_record(record=moves, file_name=self.record_file)
            return [moves]

        # Repeated mode
        history: list[list[Move]] = []
        for round_idx in range(1, self.num_rounds + 1):
            prompts = self._build_prompts_repeated(
                players, round_idx, history, similarity_pct
            )
            moves = self.base_game.play(
                additional_info=prompts, players=players
            )
            history.append(moves)

        LOGGER.log_record(record=history, file_name=self.record_file)
        return history

    @override
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Play a matchup using self.similarity_pct (for base class compatibility)."""
        return self._play_matchup_at(players, self.similarity_pct)

    # ── Tournament routing ────────────────────────────────────────────────

    @override
    def run_tournament(self, players: list[Agent]) -> PayoffsBase:
        """Run similarity tournament based on similarity_source."""
        match self.similarity_source:
            case "fixed":
                return self._run_fixed(players)
            case "sweep":
                return self._run_sweep(players)
            case "benchmark":
                return self._run_benchmark(players)
            case "benchmark_sweep":
                return self._run_benchmark_sweep(players)
            case "subjective":
                return self._run_subjective(players)
            case _:
                raise ValueError(f"Unknown similarity_source: {self.similarity_source!r}")

    # ── Mode: fixed ───────────────────────────────────────────────────────

    def _run_fixed(self, players: list[Agent]) -> PayoffsBase:
        """Run with fixed similarity percentage (existing behavior)."""
        if self.num_rounds == 1:
            return super().run_tournament(players)

        # Repeated mode: use RandomMatcher
        payoffs = self._build_payoffs()
        matcher = RandomMatcher(players)

        def process_matchup(match_up: list[Agent]) -> list[list[Move]]:
            return self._play_matchup(match_up)

        with tqdm(
            total=self.num_rounds,
            desc="Similarity rounds",
            leave=True,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix} [{desc} @ {percentage:3.0f}%]',
        ) as pbar:
            for round_idx, matches_group in enumerate(matcher, 1):
                if round_idx > self.num_rounds:
                    break
                all_matchup_results = run_tasks(
                    matches_group,
                    process_matchup,
                    max_workers=self.tournament_workers,
                )
                for matchup_rounds in all_matchup_results:
                    payoffs.add_profile(matchup_rounds)
                current_time = datetime.now().strftime('%H:%M:%S')
                pbar.set_postfix_str(f"Time: {current_time}", refresh=True)
                pbar.update(1)

        return payoffs

    # ── Mode: sweep ───────────────────────────────────────────────────────

    def _get_percentages(self) -> list[int]:
        """Get the list of similarity percentages for sweep modes."""
        if self.percentages is not None:
            return self.percentages
        pcts = list(range(0, 101, self.increment))
        if 100 not in pcts:
            pcts.append(100)
        return pcts

    def _run_sweep(self, players: list[Agent]) -> PayoffsBase:
        """Play the game at each similarity level, collecting payoffs across all levels."""
        payoffs = self._build_payoffs()
        percentages = self._get_percentages()

        # Build all player matchup pairs (same as base class)
        k = self.base_game.num_players
        players_by_id = [
            [p for p in players if p.player_id == pid]
            for pid in range(1, k + 1)
        ]
        all_pairs = self._filter_pairs(list(itertools.product(*players_by_id)))

        # Build task list: (pair, pct, trial)
        tasks = [
            (pair, pct, trial + 1)
            for pct in percentages
            for pair in all_pairs
            for trial in range(self.trials_per_level)
        ]

        total = len(tasks)
        print(f"\n[Similarity Sweep] {len(percentages)} levels × "
              f"{len(all_pairs)} pairs × {self.trials_per_level} trials = {total} games")

        # Per-level results for logging
        results_by_pct: dict[int, list] = {pct: [] for pct in percentages}

        pbar = tqdm(total=total, desc="Similarity sweep", unit="game")

        def run_single(task):
            pair, pct, trial_num = task
            result = self._play_matchup_at(pair, pct)
            pbar.update(1)
            pbar.set_postfix({"sim": f"{pct}%", "trial": trial_num})
            return pct, result

        all_results = run_tasks(tasks, run_single, max_workers=self.tournament_workers)
        pbar.close()

        for pct, match_moves in all_results:
            payoffs.add_profile(match_moves)
            results_by_pct[pct].append(
                [[{"action": m.action.to_token(),
                   "points": m.points,
                   "player": m.player.name,
                   "trace_id": m.trace_id,
                   "mix_probs": ({f"A{k}": v for k, v in m.mix_probs.items()}
                                 if m.mix_probs is not None else None)}
                  for m in round_moves]
                 for round_moves in match_moves]
            )

        # Log per-level breakdown
        LOGGER.log_record(
            {"percentages": percentages, "results_by_percentage": results_by_pct},
            "sweep_results_by_percentage.json",
        )
        return payoffs

    # ── Mode: benchmark ───────────────────────────────────────────────────

    def _run_benchmark(self, players: list[Agent]) -> PayoffsBase:
        """Run a real benchmark, compute per-pair similarity, then play the game."""
        from benchmarks.registry import (
            BENCHMARK_INFO,
            build_benchmark_prompt,
            create_benchmark,
        )

        if not self.benchmark:
            raise ValueError("benchmark param is required for similarity_source='benchmark'")

        # Phase 1: Run benchmark on each unique model (dedupe by agent_type)
        print(f"\n[Similarity Benchmark] Phase 1: Running '{self.benchmark}' benchmark...")
        bench_kwargs = dict(self.benchmark_kwargs)
        if self.benchmark_cache_dir:
            bench_kwargs["store_traces"] = True  # always store traces for cache reuse
        bench = create_benchmark(self.benchmark, **bench_kwargs)

        unique_models: dict[str, Agent] = {}
        for player in players:
            if player.agent_type not in unique_models:
                unique_models[player.agent_type] = player

        bench_results: dict[str, Any] = {}
        for model_id, agent in unique_models.items():
            cached = self._load_cached_result(model_id)
            if cached is not None:
                bench_results[model_id] = cached
            else:
                print(f"  Running benchmark on {model_id}...")
                bench_results[model_id] = bench.run(agent)
                self._save_cached_result(model_id, bench_results[model_id])

        # Map player names to their model's results for sim_matrix keying
        name_to_model: dict[str, str] = {p.name: p.agent_type for p in players}

        # Phase 2: Compute pairwise similarity
        print(f"\n[Similarity Benchmark] Phase 2: Computing pairwise similarity...")
        sim_matrix: dict[tuple[str, str], float] = {}
        agent_names = list({p.name for p in players})
        for i, name_a in enumerate(sorted(agent_names)):
            for j, name_b in enumerate(sorted(agent_names)):
                if i <= j:
                    model_a = name_to_model[name_a]
                    model_b = name_to_model[name_b]
                    if model_a == model_b:
                        sim = 100.0
                    else:
                        sim = bench.compute_similarity(
                            bench_results[model_a], bench_results[model_b]
                        )
                    sim_matrix[(name_a, name_b)] = sim
                    sim_matrix[(name_b, name_a)] = sim
                    if i != j:
                        print(f"  {name_a} vs {name_b}: {sim:.1f}%")

        LOGGER.log_record(
            {
                "benchmark": self.benchmark,
                "similarity_matrix": {
                    f"{a} vs {b}": s
                    for (a, b), s in sorted(sim_matrix.items())
                    if a <= b
                },
            },
            "benchmark_similarity.json",
        )

        # Phase 3: Play tournament with per-pair computed similarity
        print(f"\n[Similarity Benchmark] Phase 3: Playing tournament with computed similarities...")
        payoffs = self._build_payoffs()
        k = self.base_game.num_players
        players_by_id = [
            [p for p in players if p.player_id == pid]
            for pid in range(1, k + 1)
        ]
        all_pairs = self._filter_pairs(list(itertools.product(*players_by_id)))

        # Use benchmark-specific custom prompt
        original_prompt_mode = self.prompt_mode
        original_custom_prompt = self.custom_prompt
        if self.benchmark in BENCHMARK_INFO:
            self.prompt_mode = "custom"
            self.custom_prompt = build_benchmark_prompt(
                self.benchmark, difference_framing=self.difference_framing
            )

        tasks = [
            (pair, sim_matrix.get(
                (pair[0].name, pair[1].name), self.similarity_pct
            ))
            for pair in all_pairs
            for _ in range(self.trials_per_level)
        ]

        pbar = tqdm(total=len(tasks), desc="Benchmark tournament", unit="game")

        def run_single(task):
            pair, pct = task
            pct_int = int(round(pct))
            result = self._play_matchup_at(pair, pct_int)
            pbar.update(1)
            pbar.set_postfix({"sim": f"{pct_int}%"})
            return result

        all_results = run_tasks(tasks, run_single, max_workers=self.tournament_workers)
        pbar.close()

        for match_moves in all_results:
            payoffs.add_profile(match_moves)

        # Compute per-model action rates from probability distributions
        from collections import defaultdict

        action_prob_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        action_move_counts: dict[str, int] = defaultdict(int)
        actions = list(self.base_game.action_class)

        for match_moves in all_results:
            for round_moves in match_moves:
                for move in round_moves:
                    model = move.player.agent_type
                    action_move_counts[model] += 1
                    if move.mix_probs is not None:
                        for idx, prob in move.mix_probs.items():
                            action_name = actions[idx].name
                            action_prob_sums[model][action_name] += prob / 100.0
                    else:
                        action_prob_sums[model][move.action.name] += 1.0

        action_rates: dict[str, dict[str, float]] = {}
        for model in action_prob_sums:
            total = action_move_counts[model]
            action_rates[model] = {
                action: round(prob_sum / total, 4)
                for action, prob_sum in sorted(action_prob_sums[model].items())
            }

        print("\n  Action rates per model:")
        for model, rates in sorted(action_rates.items()):
            rates_str = ", ".join(f"{a}: {r:.1%}" for a, r in rates.items())
            print(f"    {model}: {rates_str}")

        LOGGER.log_record(action_rates, "action_rates.json")

        # Restore original prompt settings
        self.prompt_mode = original_prompt_mode
        self.custom_prompt = original_custom_prompt

        return payoffs

    # ── Mode: benchmark_sweep ─────────────────────────────────────────────

    def _run_benchmark_sweep(self, players: list[Agent]) -> PayoffsBase:
        """Spoofed benchmark similarity sweep: tell agents they scored X% on a benchmark.

        Sweeps across benchmarks × similarity levels, playing the game at each
        combination with benchmark-specific framing.
        """
        from benchmarks.registry import (
            BENCHMARK_INFO,
            build_benchmark_prompt_with_context,
        )

        percentages = self._get_percentages()
        benchmark_keys = self.benchmarks or list(BENCHMARK_INFO.keys())

        # Validate benchmark keys
        for bk in benchmark_keys:
            if bk not in BENCHMARK_INFO:
                raise ValueError(
                    f"Unknown benchmark: {bk!r}. "
                    f"Available: {sorted(BENCHMARK_INFO.keys())}"
                )

        payoffs = self._build_payoffs()

        # Build all player matchup pairs
        k = self.base_game.num_players
        players_by_id = [
            [p for p in players if p.player_id == pid]
            for pid in range(1, k + 1)
        ]
        all_pairs = self._filter_pairs(list(itertools.product(*players_by_id)))

        # Build task list: (pair, benchmark_key, pct, trial)
        tasks = [
            (pair, bk, pct, trial + 1)
            for bk in benchmark_keys
            for pct in percentages
            for pair in all_pairs
            for trial in range(self.trials_per_level)
        ]

        total = len(tasks)
        print(
            f"\n[Benchmark Sweep] {len(benchmark_keys)} benchmarks × "
            f"{len(percentages)} levels × {len(all_pairs)} pairs × "
            f"{self.trials_per_level} trials = {total} games"
        )

        # Per-benchmark, per-level results for logging
        results_by_benchmark: dict[str, dict[int, list]] = {
            bk: {pct: [] for pct in percentages} for bk in benchmark_keys
        }

        pbar = tqdm(total=total, desc="Benchmark sweep", unit="game")
        sems, pool_cap = self._build_model_semaphores(players)

        def run_single(task):
            pair, bk, pct, trial_num = task
            acquired = self._acquire_for_pair(sems, pair)
            try:
                # When using non-similar framing, flip the percentage:
                # e.g. 70% similar becomes 30% different/dissimilar
                flip = self.difference_framing and self.difference_framing != "similar"
                display_pct = (100 - pct) if flip else pct
                # Build prompt with ALL benchmarks listed, active one highlighted
                bench_prompt = build_benchmark_prompt_with_context(
                    bk, all_benchmarks=benchmark_keys,
                    difference_framing=self.difference_framing,
                )
                framing = build_similarity_framing(
                    similarity_pct=display_pct,
                    prompt_mode="custom",
                    custom_prompt=bench_prompt,
                    num_other_players=self.base_game.num_players - 1,
                )
                prompt = SIMILARITY_MECHANISM_PROMPT_SINGLE.format(
                    similarity_framing=framing,
                )
                prompts = [prompt] * len(pair)
                moves = self.base_game.play(
                    additional_info=prompts, players=pair
                )
                LOGGER.log_record(record=moves, file_name=self.record_file)
                pbar.update(1)
                label = "sim" if not flip else ("dissim" if self.difference_framing == "dissimilar" else "diff")
                pbar.set_postfix({"bench": bk[:8], label: f"{display_pct}%"})
                return bk, pct, [moves]
            finally:
                if sems is not None:
                    for at in acquired:
                        sems[at].release()

        # When per-model semaphores are active, ``pool_cap`` is the sum of
        # per-model caps so each model can saturate its own quota.
        all_results = run_tasks(tasks, run_single, max_workers=pool_cap)
        pbar.close()

        for bk, pct, match_moves in all_results:
            payoffs.add_profile(match_moves)
            results_by_benchmark[bk][pct].append(
                [[
                    {
                        "action": m.action.to_token(),
                        "points": m.points,
                        "player": m.player.name,
                        "trace_id": m.trace_id,
                        # Re-key mix_probs to ``A{idx}`` to match the
                        # convention the plot code expects.
                        "mix_probs": (
                            {f"A{k}": v for k, v in m.mix_probs.items()}
                            if m.mix_probs is not None else None
                        ),
                    }
                    for m in round_moves
                  ]
                 for round_moves in match_moves]
            )

        # Compute per-model cooperation rates at each benchmark × similarity level
        from collections import defaultdict

        actions = list(self.base_game.action_class)
        # Structure: {model: {benchmark: {pct: {action_name: prob_sum, "_count": n}}}}
        sweep_rates: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        )

        for bk, pct, match_moves in all_results:
            for round_moves in match_moves:
                for move in round_moves:
                    model = move.player.agent_type
                    bucket = sweep_rates[model][bk][pct]
                    bucket["_count"] += 1
                    if move.mix_probs is not None:
                        for idx, prob in move.mix_probs.items():
                            bucket[actions[idx].name] += prob / 100.0
                    else:
                        bucket[move.action.name] += 1.0

        # Convert to rates
        action_rates_by_sweep: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for model in sweep_rates:
            action_rates_by_sweep[model] = {}
            for bk in sweep_rates[model]:
                action_rates_by_sweep[model][bk] = {}
                for pct in sorted(sweep_rates[model][bk]):
                    bucket = sweep_rates[model][bk][pct]
                    n = bucket.pop("_count")
                    action_rates_by_sweep[model][bk][str(pct)] = {
                        act: round(val / n, 4) for act, val in sorted(bucket.items())
                    }

        LOGGER.log_record(
            {
                "benchmarks": benchmark_keys,
                "percentages": percentages,
                "difference_framing": self.difference_framing,
                "results_by_benchmark": results_by_benchmark,
                "action_rates_by_sweep": action_rates_by_sweep,
            },
            "benchmark_sweep_results.json",
        )
        return payoffs

    # ── Asymmetric matchup play (used by subjective mode) ─────────────────

    def _play_matchup_asymmetric(
        self,
        players: Sequence[Agent],
        per_player_pcts: list[int],
    ) -> list[list[Move]]:
        """Play a matchup where each player sees a different similarity percentage.

        Used by subjective mode where the similarity matrix is asymmetric.

        Args:
            players: The players in this matchup.
            per_player_pcts: Per-player similarity percentage (same order as players).
        """
        if self.num_rounds == 1:
            prompts = [
                SIMILARITY_MECHANISM_PROMPT_SINGLE.format(
                    similarity_framing=self._build_similarity_framing(pct),
                )
                for pct in per_player_pcts
            ]
            moves = self.base_game.play(
                additional_info=prompts, players=players
            )
            LOGGER.log_record(record=moves, file_name=self.record_file)
            return [moves]

        # Repeated mode with per-player asymmetric similarities
        history: list[list[Move]] = []
        for round_idx in range(1, self.num_rounds + 1):
            rounds_played = round_idx - 1
            prompts = []
            for player, pct in zip(players, per_player_pcts):
                if not history:
                    history_context = SIMILARITY_NO_HISTORY
                else:
                    history_lines = []
                    for idx, round_moves in enumerate(history):
                        focus_move = next(
                            (m for m in round_moves if m.player == player), None
                        )
                        if focus_move is None:
                            continue
                        actions = [
                            SIMILARITY_SELF_LABEL.format(
                                action_token=focus_move.action.to_token()
                            )
                        ]
                        for move in round_moves:
                            if move.player == player:
                                continue
                            actions.append(
                                SIMILARITY_OTHERPLAYER_LABEL.format(
                                    other_player=f"Player {move.player.player_id}",
                                    action_token=move.action.to_token(),
                                )
                            )
                        history_lines.append(
                            SIMILARITY_ROUND_LINE.format(
                                round_idx=idx + 1,
                                actions="\n".join(actions),
                            )
                        )
                    history_context = (
                        "\n".join(reversed(history_lines))
                        if history_lines
                        else SIMILARITY_NO_HISTORY
                    )

                framing = self._build_similarity_framing(pct)
                prompt = SIMILARITY_MECHANISM_PROMPT_REPEATED.format(
                    similarity_framing=framing,
                    discount=int(self.discount * 100),
                    round_idx=rounds_played,
                    history_context=history_context,
                )
                prompts.append(prompt)

            moves = self.base_game.play(
                additional_info=prompts, players=players
            )
            history.append(moves)

        LOGGER.log_record(record=history, file_name=self.record_file)
        return history

    # ── Mode: subjective ──────────────────────────────────────────────────

    def _run_subjective(self, players: list[Agent]) -> PayoffsBase:
        """Run benchmark, have agents subjectively rate each other, then play.

        Phase 1: Run benchmark on each unique agent.
        Phase 2: For each ordered pair (A,B), create a fresh instance of A,
                 show it B's responses, and ask it to rate similarity 0-100.
                 Also compute algorithmic similarity for comparison logging.
        Phase 3: Play tournament with per-player asymmetric similarities.
        """
        import numpy as np

        from benchmarks.registry import (
            BENCHMARK_INFO,
            build_benchmark_prompt,
            create_benchmark,
        )
        from src.mechanisms.subjective_similarity import SubjectiveSimilarityComputer

        if not self.benchmark:
            raise ValueError(
                "benchmark param is required for similarity_source='subjective'"
            )

        # Phase 1: Run benchmark on each unique model (dedupe by agent_type,
        # not by name which includes #P1/#P2 — same model doesn't need
        # to run the benchmark twice).
        print(
            f"\n[Subjective Similarity] Phase 1: Running '{self.benchmark}' "
            f"benchmark..."
        )
        bench_kwargs = dict(self.benchmark_kwargs)
        if self.subjective_mode in ("explanation", "both"):
            bench_kwargs["store_traces"] = True

        bench = create_benchmark(self.benchmark, **bench_kwargs)

        # Deduplicate by agent_type (model identity without player position)
        unique_models: dict[str, Agent] = {}  # agent_type -> representative agent
        model_configs: dict[str, dict] = {}   # agent_type -> config
        for player in players:
            if player.agent_type not in unique_models:
                unique_models[player.agent_type] = player
                model_configs[player.agent_type] = player.agent_config

        bench_results: dict[str, Any] = {}
        for model_id, agent in unique_models.items():
            cached = self._load_cached_result(model_id)
            if cached is not None:
                # For explanation/both, verify cache has traces
                needs_traces = self.subjective_mode in ("explanation", "both")
                has_traces = cached.raw_responses and "full_response" in cached.raw_responses[0]
                if needs_traces and not has_traces:
                    print(f"  [Cache MISS] {model_id} cached without traces, re-running...")
                    bench_results[model_id] = bench.run(agent)
                    self._save_cached_result(model_id, bench_results[model_id])
                else:
                    bench_results[model_id] = cached
            else:
                print(f"  Running benchmark on {model_id}...")
                bench_results[model_id] = bench.run(agent)
                self._save_cached_result(model_id, bench_results[model_id])

        # Phase 2: Subjective similarity computation (between unique models)
        print(
            f"\n[Subjective Similarity] Phase 2: Agents judging each other "
            f"(mode={self.subjective_mode!r})..."
        )
        computer = SubjectiveSimilarityComputer(
            mode=self.subjective_mode,
            chunk_size=self.subjective_chunk_size,
            max_items=self.subjective_max_items,
            max_workers=self.tournament_workers,
        )
        sim_matrix, all_judgments = computer.compute_subjective_matrix(
            agent_configs=model_configs,
            benchmark_results=bench_results,
        )

        # Also compute algorithmic similarity for comparison
        model_ids = list(unique_models.keys())
        algorithmic_matrix: dict[str, float] = {}
        for i, id_a in enumerate(model_ids):
            for j, id_b in enumerate(model_ids):
                if i < j:
                    algo_sim = bench.compute_similarity(
                        bench_results[id_a], bench_results[id_b]
                    )
                    algorithmic_matrix[f"{id_a} vs {id_b}"] = algo_sim

        # Log both matrices
        subjective_log: dict[str, float] = {}
        symmetric_log: dict[str, float] = {}
        for i, a in enumerate(model_ids):
            for j, b in enumerate(model_ids):
                if i < j:
                    s_ab = sim_matrix.get((a, b), 50.0)
                    s_ba = sim_matrix.get((b, a), 50.0)
                    avg = (s_ab + s_ba) / 2
                    subjective_log[f"{a} judges {b}"] = s_ab
                    subjective_log[f"{b} judges {a}"] = s_ba
                    symmetric_log[f"{a} vs {b}"] = avg
                    algo = algorithmic_matrix.get(f"{a} vs {b}", 50.0)
                    print(f"  {a} judges {b}: {s_ab:.1f}%")
                    print(f"  {b} judges {a}: {s_ba:.1f}%")
                    print(f"  Subjective avg: {avg:.1f}%  |  Algorithmic: {algo:.1f}%")

        LOGGER.log_record(
            {
                "benchmark": self.benchmark,
                "subjective_mode": self.subjective_mode,
                "asymmetric_matrix": subjective_log,
                "symmetric_averages": symmetric_log,
                "algorithmic_similarity": algorithmic_matrix,
                "judgments": [
                    {
                        "judge": j.judge_name,
                        "target": j.target_name,
                        "score": j.similarity_score,
                        "chunk_scores": j.chunk_scores,
                        "num_questions": j.num_questions_shown,
                    }
                    for j in all_judgments
                ],
            },
            "subjective_similarity.json",
        )

        # Phase 3: Play tournament with asymmetric per-player similarities
        print(
            f"\n[Subjective Similarity] Phase 3: Playing tournament with "
            f"subjective similarities..."
        )
        payoffs = self._build_payoffs()
        k = self.base_game.num_players
        players_by_id = [
            [p for p in players if p.player_id == pid]
            for pid in range(1, k + 1)
        ]
        all_pairs = list(itertools.product(*players_by_id))

        # Deduplicate pairs by model identity — e.g. with 2 models A,B
        # we get (A#P1,A#P2), (A#P1,B#P2), (B#P1,A#P2), (B#P1,B#P2)
        # but (A#P1,B#P2) and (B#P1,A#P2) are the same matchup with
        # swapped positions, so keep only one.
        seen_model_combos: set[tuple[str, ...]] = set()
        unique_pairs = []
        for pair in all_pairs:
            model_key = tuple(sorted(p.agent_type for p in pair))
            if model_key not in seen_model_combos:
                seen_model_combos.add(model_key)
                unique_pairs.append(pair)
        all_pairs = unique_pairs

        # Use benchmark-specific custom prompt if available
        original_prompt_mode = self.prompt_mode
        original_custom_prompt = self.custom_prompt
        if self.benchmark in BENCHMARK_INFO:
            self.prompt_mode = "custom"
            self.custom_prompt = build_benchmark_prompt(
                self.benchmark, difference_framing=self.difference_framing
            )

        tasks = []
        for pair in all_pairs:
            per_player_sims = []
            for player in pair:
                others = [p for p in pair if p is not player]
                # Look up similarity by model identity, not full agent name
                sims = [
                    sim_matrix.get(
                        (player.agent_type, other.agent_type),
                        self.similarity_pct,
                    )
                    for other in others
                ]
                per_player_sims.append(int(round(float(np.mean(sims)))))
            for _trial in range(self.trials_per_level):
                tasks.append((pair, per_player_sims))

        total = len(tasks)
        print(
            f"  {len(all_pairs)} unique model pairs × {self.trials_per_level} trials "
            f"= {total} games"
        )

        pbar = tqdm(
            total=total, desc="Subjective tournament", unit="game"
        )

        def run_single(task):
            pair, per_player_sims = task
            result = self._play_matchup_asymmetric(pair, per_player_sims)
            pbar.update(1)
            pbar.set_postfix(
                {"sims": [f"{s}%" for s in per_player_sims]}
            )
            return result

        all_results = run_tasks(
            tasks, run_single, max_workers=self.tournament_workers
        )
        pbar.close()

        for match_moves in all_results:
            payoffs.add_profile(match_moves)

        # Compute per-model action rates from probability distributions
        from collections import defaultdict

        action_prob_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        action_move_counts: dict[str, int] = defaultdict(int)
        actions = list(self.base_game.action_class)

        for match_moves in all_results:
            for round_moves in match_moves:
                for move in round_moves:
                    model = move.player.agent_type
                    action_move_counts[model] += 1
                    if move.mix_probs is not None:
                        for idx, prob in move.mix_probs.items():
                            action_name = actions[idx].name
                            action_prob_sums[model][action_name] += prob / 100.0
                    else:
                        action_prob_sums[model][move.action.name] += 1.0

        action_rates: dict[str, dict[str, float]] = {}
        for model in action_prob_sums:
            total = action_move_counts[model]
            action_rates[model] = {
                action: round(prob_sum / total, 4)
                for action, prob_sum in sorted(action_prob_sums[model].items())
            }

        print("\n  Action rates per model:")
        for model, rates in sorted(action_rates.items()):
            rates_str = ", ".join(f"{a}: {r:.1%}" for a, r in rates.items())
            print(f"    {model}: {rates_str}")

        LOGGER.log_record(action_rates, "action_rates.json")

        # Restore original prompt settings
        self.prompt_mode = original_prompt_mode
        self.custom_prompt = original_custom_prompt

        return payoffs
