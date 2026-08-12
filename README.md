# Similarity Mechanism for Coop Eval

This repository is the code base to the paper "Do LLMs Take Care of Their Own? Similarity Signals Can Induce Cooperation" and it started as a fork of **Coop Eval** — a framework for studying cooperation between LLM agents in game-theoretic settings — with a new **similarity mechanism**. The core research question: *does telling LLM agents they are similar to their opponent change how they cooperate?*

We add:
- A **Similarity mechanism** with multiple ways to source and communicate similarity (fixed, sweep, benchmark-based, subjective)
- A **benchmark system** for computing pairwise agent similarity from questionnaire responses
- A **similarity elicitation** pipeline that measures how agent strategies shift as told similarity varies
- A new game: **Chicken** (Hawk-Dove)

Everything else — the game abstractions, agent wrappers, and the other mechanisms and evaluation methods that ship with the framework but are not used here — comes from the original Coop Eval framework.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Similarity Mechanism](#the-similarity-mechanism)
  - [Similarity Sources](#similarity-sources)
  - [Prompt Modes](#prompt-modes)
  - [Example Configs](#example-configs)
- [Benchmarks](#benchmarks)
- [Similarity Elicitation](#similarity-elicitation)
- [New Game](#new-game)
- [Scripts](#scripts)
- [Configuration System](#configuration-system)
- [Available Games](#available-games-srcgames)
- [The Similarity Mechanism Class](#the-similarity-mechanism-class-srcmechanisms)
- [Agent Wrappers](#agent-wrappers-srcagents)
- [Concurrency Model](#concurrency-model)
- [Repository Layout](#repository-layout)
- [Output Format](#output-format)
- [Running with Inspect AI](#running-with-inspect-ai)
- [Contributing](#contributing)

---

## Installation

> Python 3.12

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
```

API keys for OpenRouter / OpenAI / Gemini are read from environment variables; see [config.py](config.py) for the expected names.

If you plan to run local Hugging Face checkpoints, point `MODEL_WEIGHTS_DIR` (in [config.py](config.py)) at a directory containing the weights.

---

## Quick Start

1. **Pick a configuration** from `configs/` or copy and edit one.
2. **Run an experiment**:

```bash
python script/run_experiment.py --config configs/main/similarity_testing.yaml --seed 42
```

3. **Inspect outputs** in `outputs/<date>/...`.

### Running on a SLURM cluster

```bash
./submit.sh          # wraps sbatch and uses run_job.sh under the hood
```

`run_job.sh` and `interactive.sh` ship with placeholders (`<your-slurm-account>`, `/path/to/your/checkout`) — fill in your account, working directory, GPU type, and walltime before submitting. Logs are written to `sbatch-logs/<year>/<month>/<day>/`.

---

## The Similarity Mechanism

The similarity mechanism (`src/mechanisms/similarity.py`) is the central contribution of this work. Before agents play a game, they are told how similar their opponent is to them. This framing can change cooperation rates dramatically.

### Similarity Sources

The `similarity_source` kwarg controls where the similarity number comes from:

| Source | What it does | Key params |
|--------|-------------|------------|
| `fixed` (default) | All pairs are told the same similarity percentage | `similarity_pct` |
| `sweep` | Play the game at each level (0%, 10%, ..., 100%) to map out the full response curve | `increment` or `percentages`, `trials_per_level` |
| `benchmark` | Run a real benchmark on both agents, compute per-pair similarity with that benchmark's metric (see [Benchmarks](#benchmarks)), then play with those values | `benchmark`, `benchmark_kwargs` |
| `benchmark_sweep` | Tell agents they scored X% on benchmark Y without actually running it (spoofed) | `benchmarks`, `increment`, `trials_per_level` |
| `subjective` | Run a benchmark, then have each agent rate the other's responses; similarity is asymmetric | `benchmark`, `subjective_mode` (`decision` / `explanation` / `both`), `chunk_size` |

### Prompt Modes

The `prompt_mode` kwarg controls how similarity is communicated to agents:

| Mode | Framing |
|------|---------|
| `percentage_updated` (default) | "The other agent's decision-making is X% similar to yours. This means that when presented with similar scenarios, your approaches and choices overlap about X% of the time." |
| `percentage` | Simple "X% similar" statement |
| `domain` | Domain-specific similarity (e.g., "risk tolerance") via `domain` kwarg |
| `vague` | Non-specific ("you are playing an opponent that has some similarity to you") |
| `construct` | States that a similarity score has been computed (on a 0–1 scale) but is not available for display — isolates the construct from any particular number |
| `custom` | Free-form text with `{similarity_pct}` placeholder |

All percentage-based modes are additionally parameterised by `difference_framing`, which flips both the wording and the number: `False`/`"similar"` shows the raw percentage as "X% similar to"; `True`/`"different"` and `"dissimilar"` show `100 − X` as "X% different from" / "X% dissimilar to". `vague` and `construct` carry no percentage, so `difference_framing` does not affect them.

For N-player games (N > 2), each opponent is listed separately with their individual similarity percentage.

The exact text of every framing, including multiplayer variants, is in [`SIMILARITY_PROMPTS_V2.md`](SIMILARITY_PROMPTS_V2.md).

### Example Configs

**Fixed similarity** (`configs/mechanisms/similarity.yaml`):
```yaml
type: Similarity
kwargs:
  similarity_source: "fixed"
  similarity_pct: 70
  prompt_mode: "percentage_updated"
```

**Similarity sweep** (`configs/mechanisms/similarity_sweep.yaml`):
```yaml
type: Similarity
kwargs:
  similarity_source: "sweep"
  prompt_mode: "percentage_updated"
  increment: 10
  trials_per_level: 10
```

**Benchmark-based** (`configs/mechanisms/similarity_benchmark.yaml`):
```yaml
type: Similarity
kwargs:
  similarity_source: "benchmark"
  benchmark: "newcomb"
  benchmark_kwargs:
    max_items: 20
  prompt_mode: "percentage_updated"
```

**Benchmark sweep** (`configs/mechanisms/similarity_benchmark_sweep.yaml`):
```yaml
type: Similarity
kwargs:
  similarity_source: "benchmark_sweep"
  benchmarks: ["newcomb", "gpqa", "cabin"]
  prompt_mode: "percentage_updated"
  increment: 10
  trials_per_level: 1
```

**Subjective similarity** (`configs/mechanisms/similarity_subjective.yaml`):
```yaml
type: Similarity
kwargs:
  similarity_source: "subjective"
  benchmark: "newcomb"
  subjective_mode: "decision"
  chunk_size: 10
  prompt_mode: "percentage_updated"
```

---

## Benchmarks

Benchmarks (`benchmarks/`) measure agent characteristics and compute pairwise similarity scores. Each benchmark implements `run(agent)` to collect responses and `compute_similarity(result_a, result_b)` to produce a 0-100% similarity score.

| Key | Name | Format | Items | Similarity Metric | Used here |
|-----|------|--------|-------|-------------------|-----------|
| `newcomb` | Newcomb-like Decision Theory | MCQ (variable options, shuffled) | 537 | Raw answer agreement | **yes** |
| `gpqa` | GPQA Diamond | 4-option MCQ (shuffled) | 198 | Cohen's kappa | — |
| `hle` | Humanity's Last Exam | MCQ or short free-text answer | 2,158 | Raw answer agreement | **yes** |
| `dilemmas` | Daily Dilemmas | Binary choice (shuffled) | 1,360 | Raw answer agreement | collected only |
| `moral_choice` | MoralChoice | Binary choice (shuffled) | 1,367 | Raw answer agreement | **yes** |
| `multi_tp` | MultiTP Trolley Problems | Binary choice (shuffled) | 460 | Cohen's kappa | — |
| `cabin` | CABIN Career Interest | 5-point Likert (Dislike → Like Very Much) | 164 | Quadratic Weighted Kappa | collected only |
| `ggb` | Greatest Good Benchmark | 7-point Likert (Strongly Disagree → Strongly Agree) | 90 | Quadratic Weighted Kappa | collected only |
| `trait` | TRAIT Personality | 4-option MCQ (shuffled) | 8,000 | Raw answer agreement | **yes** |
| `random_coin_toss` | Random Coin Toss | Comma-separated H/T sequence | 100 | Raw positional agreement | — |
| `random_coin_toss_alt` | Random Coin Toss (alt phrasing) | Comma-separated H/T sequence | 100 | Raw positional agreement | — |
| `random_die_roll` | Random Die Roll | Comma-separated 1–6 sequence | 100 | Raw positional agreement | — |
| `random_die_roll_alt` | Random Die Roll (alt phrasing) | Comma-separated 1–6 sequence | 100 | Raw positional agreement | — |
| `similarity_game` | Similarity Game | Mixed-strategy probability distributions | per config | Chance-corrected Jensen–Shannon divergence | collected only |

Item counts are the full benchmark size; `max_items` subsamples them (see [Stratified Sampling](#stratified-sampling)).

**Used here** records what this work actually ran. The four marked **yes** (`newcomb`, `trait`, `moral_choice`, `hle`) are the benchmarks behind the reported exogenous-similarity results; `cabin`, `ggb`, `dilemmas`, and `similarity_game` were collected but not carried into the game tournaments. The remainder are implemented and registered but were not run, so their metrics are untested in our setting — in particular **no reported result uses Cohen's kappa**.

**What the metrics mean**

- **Raw answer agreement** — the fraction of commonly-answered questions where both agents gave the same answer, ×100. No chance correction, so two agents that both follow a strong majority pattern score high.
- **Cohen's kappa** — agreement corrected for the agreement expected from each agent's own answer marginals: `κ = (p_o − p_e) / (1 − p_e)`, rescaled to `[0, 100]` via `(κ + 1) / 2 × 100`. 50 ≈ chance.
- **Quadratic Weighted Kappa (QWK)** — the ordinal analogue used for Likert scales, penalising disagreements by the *square* of the rating gap: `score = (1 − ½ · D_obs / D_exp) × 100`, where `D_obs` is the mean squared paired difference and `D_exp` the same under independent marginals. 100 = identical, 50 ≈ chance, 0 = maximal anti-correlation.
- **Chance-corrected JSD** — for the similarity game, the Jensen–Shannon divergence between the two agents' action distributions at matched similarity levels, normalised against a cross-level independence baseline: `κ = 1 − JSD_obs / JSD_exp`, rescaled the same way.

The LLM judge (`benchmarks/llm_judge.py`) is used for the *endogenous* similarity path (`similarity_source: "subjective"`), where each agent rates the other's responses — not for benchmark scoring, which is always one of the four metrics above.

### Stratified Sampling

When `max_items` limits the number of questions (e.g., 100), benchmarks use **stratified sampling** (`benchmarks/sampling.py`) to select questions equally across subcategories rather than taking the first N. Within each subcategory, questions are randomly sampled using a deterministic seed (default 42) for reproducibility. If a benchmark has fewer total questions than `max_items`, all questions are used. Each benchmark's subcategories are defined in its module under `benchmarks/`.

### Running benchmarks

```bash
# Run a single benchmark
python script/run_benchmarks.py --agents configs/agents/cheap_llms_3.yaml --benchmark newcomb

# Run all benchmarks
python script/run_benchmarks.py --agents configs/agents/cheap_llms_3.yaml --benchmark all

# Similarity game benchmark (requires a game config)
python script/run_benchmarks.py --agents configs/agents/two_models.yaml \
    --benchmark similarity_game --game-config main/similarity_testing.yaml
```

---

## Similarity Elicitation

The similarity elicitation pipeline (`src/mechanisms/similarity_elicitation.py`) measures how an agent's strategy distribution shifts as the told similarity changes. It does not run an actual game — it only prompts agents for their mixed strategy at each similarity level.

This is used by:
- The **similarity game benchmark** to compute Jensen-Shannon divergence between agents
- The **3-phase similarity tournament** (`script/run_similarity_tournament.py`) which elicits strategies, computes pairwise similarity from those strategies, then plays the game at the computed similarity level

```bash
# Standalone elicitation
python script/run_elicitation.py --config configs/main/similarity_elicitation.yaml

# 3-phase tournament: elicit -> compute similarity -> play
python script/run_similarity_tournament.py --config configs/main/similarity_testing.yaml
```

---

## New Game

One game was added to the Coop Eval framework for this work:

### Chicken (Hawk-Dove)

Two players simultaneously choose **Swerve** (safe) or **Dare**. If both dare, both suffer a large penalty. If one dares and the other swerves, the darer wins.

| | Swerve | Dare |
|---|--------|------|
| **Swerve** | 0, 0 | -1, 1 |
| **Dare** | 1, -1 | -10, -10 |

---

## Scripts

### Experiment runners

| Script | Purpose |
|--------|---------|
| `script/run_experiment.py` | Main entry point — loads a `configs/main/*.yaml` and runs mechanism tournament + evaluations |
| `script/run_similarity_sweep.py` | Sweep similarity from 0% to 100% and record cooperation at each level |
| `script/run_benchmark_sweep.py` | Spoofed benchmark sweep — tell agents fake similarity scores per benchmark |
| `script/run_similarity_tournament.py` | 3-phase: elicit strategies, compute pairwise similarity, play at computed level |
| `script/run_elicitation.py` | Solo/pairwise strategy elicitation under similarity framing |
| `script/run_elicitation_sweep.py` | Single-agent elicitation across similarity levels with auto-plot |
| `script/run_benchmarks.py` | Run benchmarks and compute pairwise similarity matrices |
| `script/run_fixed_point.py` | Find grounded similarity for agent pairs by sweep + weighted average |
| `script/run_random_vs_gemini.py` | Driver for the Gemini-vs-Random objective+subjective experiment matrix |
| `script/cache_benchmarks.py` | Pre-cache every benchmark for an agent config (subsequent runs hit cache) |
| `script/run_llm_judged_game.py` | Play a game and judge each move's justification with the LLM judge |
| `script/reconstruct_tournament_from_logs.py` | Rebuild a `similarity_tournament_results.json` from raw `records.jsonl` + `game_log.txt` |
| `script/log_viewer.py` | Streamlit-based browser for experiment logs |

### Plotting

All plots live in a single CLI: `script/plot.py`, as argparse subcommands. Run `python script/plot.py --help` to list them, or `python script/plot.py <subcommand> --help` for its arguments.

```bash
python script/plot.py similarity_heatmap outputs/.../benchmark_sweep_results.json
python script/plot.py similarity_tournament outputs/.../similarity_tournament_results.json
python script/plot.py newcomb outputs/.../benchmark_results.json
python script/plot.py fixed_point outputs/.../fixed_point_results.json
```

| Subcommand | What it plots |
|------------|--------------|
| `benchmark_sweep` | Per-model heatmaps of cooperation rate (benchmarks × similarity %) |
| `combined_heatmap` | Merge several elicitation_sweep.json files into a similarity × model heatmap |
| `cooperation_with_sem` | Cooperation rate vs similarity with SEM bands |
| `defection_rate` | Gemini-vs-Random comparison: similarity scores, coop/defect groups, scatter |
| `fixed_point` | Per-pair f(s) curves, fixed-point heatmap, multi-pair overlay |
| `heatmaps` | Mode × benchmark heatmaps from a `comparison.json` file |
| `newcomb` | Newcomb benchmark capabilities/EDT/CDT bars and scatter |
| `similarity_heatmap` | Heatmaps from similarity-sweep or benchmark-sweep results JSONs |
| `similarity_tournament` | Two-phase similarity tournament: told-vs-actual, payoffs, coop |
| `trust_game_sweep` | Trust-game (asymmetric) sweep: per-matchup payoff curves and aggregate |

### LLM judge

`src/llm_judge/` is a taxonomy classifier that labels game-move justifications against a configurable cooperation taxonomy (`src/llm_judge/config/cooperation_taxonomy.{py,json}`). It pairs with `benchmarks/llm_judge.py` (`LLMJudge`, `QAPair`), which the `subjective` similarity source uses to score how similar two agents' reasoning is.

Runner scripts under `script/llm_judge/`:

| Script | Purpose |
|--------|---------|
| `run_justification_judge.py` | Classify justifications from an experiment run against the taxonomy |
| `normalize_justification_labels.py` | Normalise and dedupe taxonomy labels across runs |
| `build_justification_report.py` | Build a per-mechanism / per-model report with chi-squared tests |
| `plot_taxonomy_radar.py` | Radar plot of taxonomy distribution |
| `export_taxonomy_dataset.py` | Export labelled (justification, label) pairs as a dataset |

---

## Configuration System

Configs are modular YAML files composed from four components:

```yaml
# configs/main/similarity_testing.yaml
game_config: games/prisoners_dilemma.yaml
mechanism_config: mechanisms/similarity.yaml
agents_config: agents/two_models.yaml
evaluation_config: evaluation/default_evaluation.yaml
concurrency:
  max_workers: 3
  tournament_workers: 8
```

### Agent configs (`configs/agents/`)

```yaml
# configs/agents/two_models.yaml
- llm:
    provider: OpenRouter
    model: google/gemini-3-flash-preview
    kwargs:
      temperature: 1
  type: CoTAgent
- llm:
    provider: OpenRouter
    model: openai/gpt-oss-120b
    kwargs:
      temperature: 1
  type: CoTAgent
```

Supported providers: `OpenRouter`, `OpenAI`, `Gemini`, `HFInstance` (local).
Agent types: `CoTAgent` (chain-of-thought), `IOAgent` (direct answer).

### Game configs (`configs/games/`)

```yaml
# configs/games/prisoners_dilemma.yaml
type: PrisonersDilemma
kwargs:
  payoff_matrix:
    CC: [2, 2]
    CD: [0, 3]
    DC: [3, 0]
    DD: [1, 1]
```

Add `ordinal_payoffs: true` to describe outcomes to agents as preference orderings ("the outcome you prefer the most") instead of point values, keeping the ranking but hiding the magnitudes. Supported by `PrisonersDilemma`, `StagHunt`, `Chicken`, and `MatchingPennies`; see `configs/games/prisoners_dilemma_ordinal.yaml`.

### Mechanism configs (`configs/mechanisms/`)

See [The Similarity Mechanism](#the-similarity-mechanism) for the similarity configs used in this work.

---

## Available Games (`src/games/`)

| Class | Description | Origin |
|-------|-------------|--------|
| `PrisonersDilemma` | Two-player PD with configurable payoff matrix | Coop Eval |
| `PublicGoods` | N-player public goods contribution with multiplier | Coop Eval |
| `TravellersDilemma` | Two-player traveller's dilemma parameterised by min claim, spacing, bonus | Coop Eval |
| `TrustGame` | Two-player simultaneous trust game (invest vs. keep) | Coop Eval |
| `StagHunt` | Two-player stag hunt coordination game | Coop Eval |
| `MatchingPennies` | Two-player zero-sum matching pennies game | Coop Eval |
| `Chicken` | Two-player game of chicken (hawk-dove) | **New** |

---

## The Similarity Mechanism Class (`src/mechanisms/`)

| Class | Config `type` | Purpose |
|-------|---------------|---------|
| `Similarity` | `Similarity` | Tells agents about opponent similarity; multiple sources and framings |

Coop Eval's other mechanisms (reputation, mediation, disarmament, contracting, repetition) remain in the tree and are documented upstream; they are not used in this work.

---

## Agent Wrappers (`src/agents/`)

- `IOAgent`: direct answer style (no extra reasoning instructions).
- `CoTAgent`: appends "think step by step" prompts to encourage chain-of-thought.
- Backends provided by `LLMManager`:
  - `HFInstance` (local Hugging Face checkpoints, with automatic device placement).
  - `ClientAPILLM` (OpenAI-compatible API clients: OpenAI, Gemini, OpenRouter). Configure API keys in `config.py` / environment variables.

---

## Concurrency Model

- **Games** share a `_collect_actions` helper that can prompt agents either sequentially or in parallel (`parallel_players=True`).
- **Mechanisms** use a common `run_tasks` helper (`src/utils/concurrency.py`) to fan out matchups in parallel.
- Seat cloning (`Agent.make_seat_clone`) produces human-friendly labels like `Gemma(CoT)#2`, preventing name collisions when identical models face off.

---

## Repository Layout

```
.
├── benchmarks/              # Benchmark implementations + registry (new)
│   ├── registry.py          # Factory for all benchmarks
│   ├── newcomb.py, gpqa.py, hle.py, ...
│   └── data/                # Question datasets (JSON/CSV)
├── configs/
│   ├── agents/              # Agent configurations
│   ├── games/               # Game payoff matrices
│   ├── main/                # Top-level experiment configs
│   ├── mechanisms/          # Mechanism parameters
│   └── evaluation/          # Evaluation settings (unused in this work)
├── script/
│   ├── run_experiment.py    # Main entry point
│   ├── run_similarity_sweep.py
│   ├── run_similarity_tournament.py
│   ├── run_benchmarks.py
│   ├── run_benchmark_sweep.py
│   ├── run_elicitation.py
│   ├── run_elicitation_sweep.py
│   ├── run_fixed_point.py
│   ├── run_random_vs_gemini.py
│   ├── run_llm_judged_game.py
│   ├── cache_benchmarks.py
│   ├── reconstruct_tournament_from_logs.py
│   ├── log_viewer.py
│   ├── plot.py              # Unified plotting CLI (subcommands per plot)
│   └── llm_judge/           # LLM-as-judge runners (classify, normalize, plot, report)
├── inspect_similarity/      # Inspect AI integration layer (new)
│   ├── agents/              # InspectAgent adapter (Agent ABC → Inspect Model)
│   ├── tasks/               # @task definitions (tournament, benchmark, elicitation)
│   └── plotting/            # Extract data from Inspect logs for plots
├── src/
│   ├── agents/              # Agent abstractions & LLM backends
│   ├── games/               # Game definitions (+ chicken.py)
│   ├── mechanisms/          # Incentive layers (+ similarity.py, similarity_elicitation.py, subjective_similarity.py)
│   ├── llm_judge/           # Cooperation-taxonomy classifier (judge, taxonomy, processors)
│   ├── registry/            # Game, mechanism, agent registries
│   ├── evolution/           # Replicator dynamics
│   └── utils/               # Concurrency, visualization helpers
├── outputs/                 # Timestamped experiment logs
├── data/                    # Cached similarity data
├── run_job.sh, submit.sh    # SLURM job scripts
└── README.md
```

---

## Output Format

Each experiment run produces a timestamped directory under `outputs/<year>/<month>/<day>/<time>/` containing:

| File | Contents |
|------|----------|
| `game_log.txt` | Full prompt/response transcript for every agent interaction |
| `config.json` | The effective configuration used for this run |
| `matchup_payoffs.json` | Aggregated payoff table by agent matchup |
| `agent_average_payoff.json` | Average payoffs per agent |
| `benchmark_results_*.json` | Per-benchmark raw results (if benchmarks were run) |
| `similarity_computation.json` | Pairwise similarity scores (if tournament mode) |

---

## Running with Inspect AI

The project can also be run through [Inspect AI](https://inspect.aisi.org.uk/), the UK AI Safety Institute's evaluation framework. This gives you Inspect's model abstraction (unified access to OpenAI, Anthropic, Google, HuggingFace, OpenRouter), structured logging, and the `inspect view` web dashboard — while all existing game, mechanism, and benchmark logic runs unchanged.

### Setup

```bash
uv pip install inspect-ai
```

### Running experiments

**Tournament**:
```bash
# Prisoner's Dilemma with similarity sweep
inspect eval inspect_similarity/tasks/tournament.py \
    -T game=PrisonersDilemma \
    -T mechanism=Similarity \
    -T mechanism_kwargs='{"similarity_source":"sweep","prompt_mode":"percentage_updated"}' \
    -T agents_config=agents/cheap_llms_3.yaml \
    -T seed=42

# Another game, fixed similarity instead of a sweep
inspect eval inspect_similarity/tasks/tournament.py \
    -T game=StagHunt \
    -T mechanism=Similarity \
    -T mechanism_kwargs='{"similarity_source":"fixed","similarity_pct":70}' \
    -T agents_config=agents/two_models.yaml \
    -T seed=42
```

**From existing YAML config** (auto-detects tournament vs elicitation):
```bash
inspect eval inspect_similarity/tasks/from_config.py \
    -T config_path=main/similarity_testing.yaml \
    -T seed=42

# With benchmark override
inspect eval inspect_similarity/tasks/from_config.py \
    -T config_path=main/similarity_testing.yaml \
    -T benchmark=newcomb \
    -T max_items=20 \
    -T seed=42
```

**Benchmarks**:
```bash
# Single benchmark
inspect eval inspect_similarity/tasks/benchmark_task.py@benchmark_eval \
    -T benchmark_name=newcomb \
    -T agents_config=agents/cheap_llms_3.yaml \
    -T max_items=20

# Battery of benchmarks
inspect eval inspect_similarity/tasks/benchmark_task.py@benchmark_battery \
    -T benchmarks=newcomb,cabin,dilemmas,moral_choice \
    -T agents_config=agents/cheap_llms_3.yaml \
    -T max_items=20
```

**Elicitation** (strategy distributions at different similarity levels):
```bash
inspect eval inspect_similarity/tasks/elicitation.py \
    -T game=PrisonersDilemma \
    -T agents_config=agents/two_models.yaml \
    -T similarity_pct=70 \
    -T prompt_mode=percentage_updated \
    -T mode=pairwise \
    -T seed=42
```

### Viewing results

```bash
inspect view
```

This opens a web dashboard at `localhost:7575` showing all eval runs, per-sample results, and full message transcripts.

### Extracting data for plots

```python
from inspect_similarity.plotting.from_logs import (
    load_tournament_results,
    load_benchmark_results,
    load_elicitation_results,
    load_similarity_matrix,
)

# From an Inspect eval log
tournament = load_tournament_results("logs/2026-04-08T.../eval.json")
print(tournament["agent_average_payoff"])

benchmarks = load_benchmark_results("logs/2026-04-08T.../eval.json")
matrix = load_similarity_matrix("logs/2026-04-08T.../eval.json")
```

### How it works

The Inspect integration is a thin wrapper — it does **not** rewrite any game, mechanism, or benchmark logic. Instead:

1. **`InspectAgent`** (`inspect_similarity/agents/inspect_agent.py`) subclasses the existing `Agent` ABC but routes LLM calls through Inspect's `Model.generate()` instead of the custom `ClientAPILLM` stack.
2. **Inspect tasks** (`inspect_similarity/tasks/`) are thin entry points that create `InspectAgent` instances and call the existing pipeline functions (`mechanism.run_tournament()`, `benchmark.run()`, `elicitation.elicit_single()`, etc.).
3. **Results** are stored in both Inspect's structured log system and the existing `outputs/` directory via `LOGGER`.

### Package layout

```
inspect_similarity/
├── agents/
│   └── inspect_agent.py       # InspectAgent, InspectCoTAgent, InspectIOAgent
├── tasks/
│   ├── tournament.py           # @task: full tournament pipeline
│   ├── benchmark_task.py       # @task: benchmarks + pairwise similarity
│   ├── elicitation.py          # @task: similarity elicitation
│   └── from_config.py          # @task: load existing YAML configs
└── plotting/
    └── from_logs.py            # Extract data from Inspect eval logs
```

---

## Contributing

This repository is focused on the similarity mechanism, so the most useful contributions are ones that extend it:

- **Add a benchmark** — the main lever for exogenous similarity. Subclass `Benchmark` in `benchmarks/`, implement `run()` and `compute_similarity()`, and register it in `benchmarks/registry.py`. A benchmark is only as useful as its similarity metric, so state plainly what `compute_similarity` returns and whether it is chance-corrected.
- **Add a similarity framing** — a new `prompt_mode` in `src/mechanisms/prompts.py`, dispatched from `similarity_utils.py`. Please add the exact wording to [`SIMILARITY_PROMPTS_V2.md`](SIMILARITY_PROMPTS_V2.md) in the same format as the existing entries, since that file is the reference for what agents actually saw.
- **Add a similarity source** — a new `similarity_source` branch in `src/mechanisms/similarity.py`, for a different way of deriving the number the agents are told.
- **Report a result that does not replicate** — cooperation under similarity framing is sensitive to model, payoff scale, and prompt wording. Runs that disagree with ours are useful; please include the config and the model versions.

If instead you have a **new game or a new mechanism** that is not specific to similarity, it belongs upstream in [Coop Eval](https://github.com/Akash190104/CoopEval) rather than here — that is where the game abstractions, the mechanism interface, and the other mechanisms live, and a contribution there benefits every mechanism built on top of it, this one included.
