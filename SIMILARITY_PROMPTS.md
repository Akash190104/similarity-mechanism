# All Similarity Prompts

This document catalogues every prompt shown to LLM agents related to the similarity mechanism.

---

## 1. Wrapper Prompts (how similarity framing is injected)

### Single-shot game wrapper
**Source:** `src/mechanisms/prompts.py`

```
Here is the twist:
{similarity_framing}
```

### Repeated game wrapper
**Source:** `src/mechanisms/prompts.py`

```
Here is the twist:
You are playing this game *repeatedly* with varying players who you encounter at random.
{similarity_framing}
The action sampled from your action probability distribution will be visible to future opponents.
After each round, there is a {discount}% chance probability that an additional round will take place.
You have already played this game for {round_idx} round(s) in the past.

{history_context}
```

---

## 2. Similarity Framing Variants (what fills `{similarity_framing}`)

All percentage-based framings below support the `difference_framing` config
(see table at the end of section 3 for `{measure_word}` / `{relation_word}` values
and percentage-flip behaviour).

### `prompt_mode="percentage"`
**2-player:**
```
Your opponent is {similarity_pct}% {measure_word} you. Remember, the other agent is seeing this information as well.
```

**Multiplayer:**
```
Each of the other {num_other_players} agents is {similarity_pct}% {measure_word} you. Remember, the other agents are seeing this information as well.
```

### `prompt_mode="percentage_updated"` (default)
**2-player:**
```
The other agent's decision-making is {similarity_pct}% {measure_word} yours, meaning, this is how {relation_word} you and the other agent reason and come to conclusions when facing the same strategic problem. Note, however, you and the other agent are independently trying to maximize your own total points. Remember, the other agent is seeing this information as well.
```

**Multiplayer:**
```
Each of the other {num_other_players} agents' decision-making is {similarity_pct}% {measure_word} yours, meaning, this is how {relation_word} you and the other agents reason and come to conclusions when facing the same strategic problem. Note, however, you and the other agents are independently trying to maximize your own total points. Remember, the other agents are seeing this information as well.
```

### `prompt_mode="domain"`
**2-player:**
```
Your opponent is {similarity_pct}% {measure_word} you in the domain of {domain}. Remember, the other agent is seeing this information as well.
```

**Multiplayer:**
```
Each of the other {num_other_players} agents is {similarity_pct}% {measure_word} you in the domain of {domain}. Remember, the other agents are seeing this information as well.
```

### `prompt_mode="vague"`
```
You are playing an opponent that has some similarity to you. Remember, the other agent is seeing this information as well.
```

### `prompt_mode="custom"`
```
{custom_text}
```

---

## 3. Benchmark-Based Framing

### Single benchmark (`similarity_source="benchmark"`)
**Source:** `benchmarks/registry.py` — `build_benchmark_prompt()`

```
Before this game, both you and the other agent were independently given the {benchmark_full_name} -- {benchmark_description}. The benchmark format is: {benchmark_format}. Example questions from the benchmark: {benchmark_sample} Based on your respective answers, the other agent's decision-making is {similarity_pct}% {measure_word} yours, meaning, this is how {relation_word} you and the other agent reason and come to conclusions when facing the same strategic problem. Note, however, you and the other agent are independently trying to maximize your own total points. Think hard about how important this benchmark is for your decision-making. Remember, the other agent is seeing this information as well.
```

### Benchmark sweep with full context (`similarity_source="benchmark_sweep"`)
**Source:** `benchmarks/registry.py` — `build_benchmark_prompt_with_context()`

```
Before this game, both you and the other agent were independently given a battery of benchmarks to measure similarities/differences. Here is the full list of benchmarks you both completed:

{catalogue}

For this game, the benchmark being used to measure your similarities/differences is the {active_benchmark_full_name} ({benchmark_key}). Based on your respective answers to this benchmark, the other agent's decision-making is {similarity_pct}% {measure_word} yours, meaning, this is how {relation_word} you and the other agent reason and come to conclusions when facing the same strategic problem. Note, however, you and the other agent are independently trying to maximize your own total points. Think hard about how important this benchmark is for your decision-making. Remember, the other agent is seeing this information as well.
```

**Framing variations** (controlled by `difference_framing` config — applies to all percentage-based prompts throughout this doc):

| `difference_framing` | `{measure_word}` | `{relation_word}` | Percentage shown |
|---|---|---|---|
| `False` or `"similar"` | similar to | similar | raw similarity % |
| `True` or `"different"` | different from | different | 100 - similarity % |
| `"dissimilar"` | dissimilar to | dissimilar | 100 - similarity % |

**Catalogue format** (one entry per benchmark):
```
1. {full_name} ({key}): {description}. Format: {format}.
```

---

## 4. Multiplayer Custom Framing (built in `similarity.py`)

When `num_other_players > 1` in benchmark modes, a per-player framing is constructed.
The `{measure_word}` / `{relation_word}` slots and flipped percentage behave as in section 3.

```
The following describes how {relation_word} each other player's decision-making is to yours:
- Player {player_id}'s decision-making is {similarity_pct}% {measure_word} yours.
- Player {player_id}'s decision-making is {similarity_pct}% {measure_word} yours.
...

This means how {relation_word} you and each other player reason and come to conclusions when facing the same strategic problem. Note, however, all players are independently trying to maximize their own total points.
```

---

## 5. Subjective Similarity Prompts (agent self-assesses similarity)

These prompts are shown to agents *before* the game. The agent reads the other agent's benchmark responses and produces a similarity score, which then gets used as the `{similarity_pct}`.

### `subjective_mode="decision"` (answers only, no reasoning)
**Source:** `src/mechanisms/subjective_similarity.py`

```
You are about to play a strategic game against another agent. Before the game, both you and the other agent were independently given a set of questions. Below are the other agent's responses to those questions. You do NOT see your own responses here -- only theirs.

Based on these responses, assess how similar the other agent's decision-making style is to your own. Consider:
- Do their answers suggest they would reach similar conclusions as you?
- Do they seem to apply similar reasoning strategies as you would?
- Do they show similar preferences or biases as you?

{dossier}

Provide a similarity score from 0 to 100, where:
- 0 means their decision-making is completely different from yours
- 50 means moderately similar to yours
- 100 means nearly identical to your decision-making style

Think step by step about what their answers reveal about their decision-making, compare it to how you would approach the same problems, and then provide your final score.

Your response MUST end with exactly: SIMILARITY SCORE: <number>
```

### `subjective_mode="explanation"` (reasoning only, answers redacted)

```
You are about to play a strategic game against another agent. Before the game, both you and the other agent were independently given a set of questions. Below are the other agent's reasoning processes for those questions. Their final answers have been redacted -- you can only see how they think, not what they concluded. You do NOT see your own responses here -- only theirs.

Based on their reasoning, assess how similar the other agent's decision-making style is to your own. Consider:
- Do they follow similar chains of reasoning as you would?
- Do they weigh similar factors when making decisions?
- Do they show similar analytical approaches as you?
- Do their thought processes suggest similar biases or preferences as yours?

{dossier}

Provide a similarity score from 0 to 100, where:
- 0 means their reasoning style is completely different from yours
- 50 means moderately similar to yours
- 100 means nearly identical to your reasoning style

Think step by step about what their reasoning reveals about their decision-making process, compare it to how you would approach the same problems, and then provide your final score.

Your response MUST end with exactly: SIMILARITY SCORE: <number>
```

### `subjective_mode="both"` (reasoning + answers)

```
You are about to play a strategic game against another agent. Before the game, both you and the other agent were independently given a set of questions. Below are the other agent's reasoning processes and final answers to those questions. You do NOT see your own responses here -- only theirs.

Based on their reasoning and answers, assess how similar the other agent's decision-making style is to your own. Consider:
- Do they follow similar chains of reasoning as you would?
- Do they reach similar conclusions as you?
- Do they weigh similar factors when making decisions?
- Do they show similar analytical approaches, preferences, or biases as you?

{dossier}

Provide a similarity score from 0 to 100, where:
- 0 means their decision-making is completely different from yours
- 50 means moderately similar to yours
- 100 means nearly identical to your decision-making style

Think step by step about what their reasoning and answers reveal about their decision-making, compare it to how you would approach the same problems, and then provide your final score.

Your response MUST end with exactly: SIMILARITY SCORE: <number>
```

---

## 6. History Context (for repeated games)

### No history yet
```
No rounds have been played yet.
```

### Per-round history format
```
[Round {round_idx}]
	You: {action_token}
	{other_player}: {action_token}
```

---

## 7. Quick Reference: When Each Prompt Is Used

| Similarity Source | Prompt Mode | What the agent sees |
|---|---|---|
| `fixed` | `percentage_updated` (default) | Wrapper + percentage_updated framing with fixed % |
| `sweep` | `percentage_updated` | Wrapper + percentage_updated framing, % varies across runs |
| `benchmark` | `custom` (auto) | Wrapper + single benchmark framing |
| `benchmark_sweep` | `custom` (auto) | Wrapper + benchmark sweep framing with catalogue |
| `subjective` | `custom` (auto) | Agent first sees subjective prompt, then wrapper + percentage_updated with self-assessed % |
