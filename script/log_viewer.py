"""Streamlit-based game log viewer for browsing and filtering experiment logs.

Run with:
    streamlit run script/log_viewer.py

Inputs:
    Recursively scans `outputs/`, `results/`, and `data/` under the project
    root for `game_log.txt` files (or accepts a custom path), and reads the
    sibling `config.json` when present. Read-only — produces no files.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from script.log_parser import LogPair, extract_metadata, pair_entries, parse_game_log


def find_log_files(root: Path) -> list[Path]:
    """Recursively find all game_log.txt files under root."""
    return sorted(root.rglob("game_log.txt"))


def load_config_for_log(log_path: Path) -> dict | None:
    """Try to load config.json from the same directory as the log file."""
    config_path = log_path.parent / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def filter_pairs(
    pairs: list[LogPair],
    *,
    selected_models: list[str] | None = None,
    selected_agents: list[str] | None = None,
    round_range: tuple[int, int] | None = None,
    similarity_range: tuple[int, int] | None = None,
    search_text: str = "",
) -> list[LogPair]:
    """Filter log pairs based on criteria."""
    filtered = pairs

    if selected_models:
        filtered = [p for p in filtered if p.model in selected_models]

    if selected_agents:
        filtered = [p for p in filtered if p.agent in selected_agents]

    if round_range:
        min_r, max_r = round_range
        filtered = [
            p for p in filtered
            if p.round_number is None or min_r <= p.round_number <= max_r
        ]

    if similarity_range:
        min_s, max_s = similarity_range
        filtered = [
            p for p in filtered
            if p.similarity_pct is None or min_s <= p.similarity_pct <= max_s
        ]

    if search_text:
        search_lower = search_text.lower()
        filtered = [
            p for p in filtered
            if search_lower in p.prompt.lower() or search_lower in p.response.lower()
        ]

    return filtered


def main() -> None:
    st.set_page_config(
        page_title="Game Log Viewer",
        page_icon="🎮",
        layout="wide",
    )

    st.title("Game Log Viewer")
    st.markdown("Browse and filter game experiment logs.")

    # --- Sidebar: File Selection ---
    st.sidebar.header("Log File Selection")

    # Find log files in common directories
    project_root = Path(__file__).resolve().parent.parent
    search_dirs = [
        project_root / "outputs",
        project_root / "results",
        project_root / "data",
    ]

    all_log_files: list[Path] = []
    for d in search_dirs:
        if d.exists():
            all_log_files.extend(find_log_files(d))

    # Custom path input
    custom_path = st.sidebar.text_input(
        "Or enter custom log file path:",
        placeholder="/path/to/game_log.txt",
    )

    if custom_path:
        custom = Path(custom_path)
        if custom.exists() and custom.is_file():
            all_log_files = [custom] + all_log_files
        else:
            st.sidebar.error(f"File not found: {custom_path}")

    if not all_log_files:
        st.warning("No game_log.txt files found. Check your outputs/results/data directories.")
        return

    # Format paths for display (relative to project root)
    def display_path(p: Path) -> str:
        try:
            return str(p.relative_to(project_root))
        except ValueError:
            return str(p)

    log_options = {display_path(p): p for p in all_log_files}
    selected_display = st.sidebar.selectbox(
        "Select log file:",
        options=list(log_options.keys()),
    )

    if not selected_display:
        return

    selected_path = log_options[selected_display]

    # --- Parse Log ---
    with st.spinner("Parsing log file..."):
        entries = parse_game_log(selected_path)
        pairs = pair_entries(entries)
        metadata = extract_metadata(pairs)

    # Show config info if available
    config = load_config_for_log(selected_path)
    if config:
        with st.sidebar.expander("Experiment Config"):
            game_type = config.get("game", {}).get("type", "Unknown")
            mechanism_type = config.get("mechanism", {}).get("type", "Unknown")
            st.write(f"**Game:** {game_type}")
            st.write(f"**Mechanism:** {mechanism_type}")
            if "name" in config:
                st.write(f"**Name:** {config['name']}")

    # --- Sidebar: Filters ---
    st.sidebar.header("Filters")

    # Model filter
    selected_models = None
    if metadata["models"]:
        selected_models = st.sidebar.multiselect(
            "Model:",
            options=metadata["models"],
            default=None,
        )
        if not selected_models:
            selected_models = None

    # Agent filter
    selected_agents = None
    if metadata["agents"]:
        selected_agents = st.sidebar.multiselect(
            "Agent (model + player position):",
            options=metadata["agents"],
            default=None,
        )
        if not selected_agents:
            selected_agents = None

    # Round filter
    round_range = None
    if metadata["round_numbers"]:
        round_range = st.sidebar.slider(
            "Round number range:",
            min_value=metadata["min_round"],
            max_value=metadata["max_round"],
            value=(metadata["min_round"], metadata["max_round"]),
        )

    # Similarity filter
    similarity_range = None
    if metadata["has_similarity"]:
        min_sim = min(metadata["similarity_values"])
        max_sim = max(metadata["similarity_values"])
        similarity_range = st.sidebar.slider(
            "Similarity score range:",
            min_value=min_sim,
            max_value=max_sim,
            value=(min_sim, max_sim),
        )

    # Text search
    search_text = st.sidebar.text_input(
        "Search in prompt/response:",
        placeholder="e.g., cooperate, defect, A0",
    )

    # --- Apply Filters ---
    filtered = filter_pairs(
        pairs,
        selected_models=selected_models,
        selected_agents=selected_agents,
        round_range=round_range,
        similarity_range=similarity_range,
        search_text=search_text,
    )

    # --- Main View ---
    st.markdown(f"**Showing {len(filtered)} of {len(pairs)} entries**")
    st.markdown(f"*Log file: `{selected_display}`*")

    if not filtered:
        st.info("No entries match the current filters.")
        return

    # Pagination
    page_size = st.sidebar.number_input("Entries per page:", min_value=5, max_value=100, value=20)
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.sidebar.number_input("Page:", min_value=1, max_value=total_pages, value=1)

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered))
    page_pairs = filtered[start_idx:end_idx]

    st.markdown(f"*Page {page} of {total_pages}*")

    # Display entries
    for pair in page_pairs:
        # Header
        header_parts = [f"**{pair.agent}**", f"`{pair.trace_id}`", pair.timestamp]
        if pair.round_number is not None:
            header_parts.append(f"Round {pair.round_number}")
        if pair.similarity_pct is not None:
            header_parts.append(f"Similarity: {pair.similarity_pct}%")

        st.markdown(" | ".join(header_parts))

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Prompt", expanded=False):
                st.markdown(pair.prompt)

        with col2:
            with st.expander("Response", expanded=True):
                st.markdown(pair.response)

        st.divider()


if __name__ == "__main__":
    main()
