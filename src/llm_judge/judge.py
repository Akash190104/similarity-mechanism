"""
Core LLM Judge functionality for text classification using custom taxonomies.
"""

import json
import textwrap
from typing import Any

from src.agents.client_api_llm import ClientAPILLM


class LLMJudge:
    """
    Main class for performing LLM-based text classification.
    """

    def __init__(
        self, api_client: ClientAPILLM, taxonomy, temperature: float = 0
    ):
        """
        Initialize the LLM Judge.

        Args:
            api_client: API client instance (OpenAI-compatible client)
            taxonomy: Taxonomy instance with categories and definitions
            temperature: Temperature for LLM generation (0 = deterministic)
        """
        self.api_client = api_client
        self.taxonomy = taxonomy
        self.temperature = temperature

    def classify_text(self, text: str, max_tokens: int = 900) -> dict[str, Any]:
        """
        Classify a single text using the LLM judge.

        Args:
            text: Text to classify
            max_tokens: Maximum tokens for LLM response

        Returns:
            Dictionary with classification results
        """
        prompt = self._build_classification_prompt(text)

        try:
            response = self.api_client.invoke(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )

            json_result = self._extract_json_from_response(response)

            if json_result:
                parsed = json.loads(json_result)
                return self._build_result(parsed)
            else:
                return {
                    "Reasoning_behind_classification": response,
                    "Confidence": 0.0,
                    "category_assignments": {},
                    "justification_type": "Failed classification",
                }

        except json.JSONDecodeError as exc:
            return {
                "Reasoning_behind_classification": f"Invalid JSON in response: {exc}",
                "Confidence": 0.0,
                "category_assignments": {},
                "justification_type": "Failed classification",
            }

    def _build_result(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Validate parsed JSON into the canonical per-category result shape."""
        reasoning = str(parsed.get("Reasoning_behind_classification", ""))
        try:
            confidence = float(parsed.get("Confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        raw_assignments = parsed.get("category_assignments")
        category_keys = self.taxonomy.get_categories()

        if not isinstance(raw_assignments, dict) or not raw_assignments:
            return {
                "Reasoning_behind_classification": reasoning,
                "Confidence": confidence,
                "category_assignments": {},
                "justification_type": "Failed classification",
            }

        validated: dict[str, bool] = {}
        for key in category_keys:
            value = raw_assignments.get(key)
            if value is True or value is False:
                validated[key] = value
            else:
                # Strict mode: any non-bool value invalidates the response.
                return {
                    "Reasoning_behind_classification": reasoning,
                    "Confidence": confidence,
                    "category_assignments": {},
                    "justification_type": "Failed classification",
                }

        true_labels = [k for k in category_keys if validated[k]]
        justification_type = ", ".join(true_labels) if true_labels else "Others"

        return {
            "Reasoning_behind_classification": reasoning,
            "Confidence": confidence,
            "category_assignments": validated,
            "justification_type": justification_type,
        }

    def _build_classification_prompt(self, text: str) -> str:
        """Build the classification prompt using the taxonomy."""

        taxonomy_text = self.taxonomy.get_formatted_taxonomy()
        category_keys = self.taxonomy.get_categories()
        n_categories = len(category_keys)

        skeleton_lines = [
            f'    "{key}": <true|false>' for key in category_keys
        ]
        skeleton = ",\n".join(skeleton_lines)

        prompt = textwrap.dedent("""\
            Analyze the following text (a model's reasoning trace from a game-theoretic decision)
            and INDEPENDENTLY evaluate whether each of the categories below is reflected.

            {taxonomy}

            Text to analyze:
            \"\"\"
            {text}
            \"\"\"

            INSTRUCTIONS:
            1. Treat each of the {n} categories as a SEPARATE yes/no question. Do NOT pick "the
               best" category — multiple categories typically apply at once.
            2. Output true if the trace shows ANY evidence of that consideration, even briefly.
               Output false only if there is no evidence.
            3. Picking only one true is almost always wrong.
            4. Use "Others" = true ONLY if the reasoning genuinely fits none of the other categories.

            Return EXACTLY this JSON, no extra prose:
            {{
              "Reasoning_behind_classification": "<2-4 sentences explaining which categories applied and why>",
              "Confidence": <decimal between 0 and 1>,
              "category_assignments": {{
            {skeleton}
              }}
            }}

            REQUIREMENTS:
            - "category_assignments" MUST contain EXACTLY one boolean per category — all {n} keys
              present, no extras, no nulls, no strings.
            - "Confidence" is a decimal number between 0 and 1, not a string.
            - No prose outside the JSON object.
            """)
        return prompt.format(
            taxonomy=taxonomy_text,
            text=text,
            n=n_categories,
            skeleton=skeleton,
        )

    @staticmethod
    def _extract_json_snippet(text: str) -> str | None:
        """Slice and validate the first JSON object embedded in ``text``."""
        if not text:
            return None

        brace_start = text.find("{")
        if brace_start == -1:
            return None

        decoder = json.JSONDecoder()
        try:
            _, end = decoder.raw_decode(text[brace_start:])
        except json.JSONDecodeError:
            return None

        return text[brace_start : brace_start + end]

    def _extract_json_from_response(self, response: str) -> str | None:
        """Return the JSON payload found inside an LLM response, if any."""

        snippet = self._extract_json_snippet(response)
        if snippet is not None:
            return snippet

        normalized = response.replace("'", '"')
        if normalized == response:
            return None

        return self._extract_json_snippet(normalized)
