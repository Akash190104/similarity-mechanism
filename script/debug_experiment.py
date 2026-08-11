#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Specify Configs and simulate command line arguments
from script.run_experiment import main, set_seed

CONFIG = "main/similarity_testing.yaml"
# CONFIG = "main/pd_testing.yaml"
# CONFIG = "main/pg_reputation_test.yaml"
sys.argv = ["script/run_experiment.py", "--config", CONFIG, "--seed", "42"]

# ============ Legacy Configs for reference ============
# from script.run_evolution import set_seed, main
# CONFIG = "legacy/tg_testing_openrouter.yaml"
# sys.argv = ["run_evolution.py", "--config", CONFIG]


# Run the main script
if __name__ == "__main__":
    set_seed()
    output_dir = main()

    # Run analysis pipeline on the output
    if output_dir:
        from data_analysis.pipeline import analyze

        analysis_dir = output_dir / "analysis"
        analyze(output_dir, analysis_dir)
