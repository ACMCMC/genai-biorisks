"""
Run predefined experimental configurations
"""

import argparse
import subprocess
import sys
import yaml
import os


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(config_name: str, config_path: str):
    """Run experiment with configuration from YAML file"""

    print(f"\n{'='*50}")
    print(f"RUNNING {config_name.upper()} EXPERIMENT")
    print(f"{'='*50}")

    # Build command
    cmd = [
        sys.executable,
        "main_pipeline.py",
        "--config",
        config_path,
    ]

    print("Command:", " ".join(cmd))
    print()

    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {config_name.upper()} EXPERIMENT COMPLETED SUCCESSFULLY")
        
        # Load config to show results files
        config = load_config(config_path)
        return config

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {config_name.upper()} EXPERIMENT FAILED")
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run predefined experiments")
    parser.add_argument(
        "experiment",
        choices=["small", "medium", "full", "comparison"],
        help="Experiment configuration to run",
    )
    parser.add_argument(
        "--use-existing-data",
        action="store_true",
        help="Use existing data if available",
    )

    args = parser.parse_args()

    # Map experiment names to config files
    config_files = {
        "small": "configs/small_scale.yaml",
        "medium": "configs/medium_scale.yaml", 
        "full": "configs/full_scale.yaml",
        "comparison": "configs/model_comparison.yaml",
    }

    config_path = config_files[args.experiment]

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    # Add existing data flag by modifying the config temporarily if needed
    if args.use_existing_data:
        print(f"\n{'='*50}")
        print(f"RUNNING {args.experiment.upper()} EXPERIMENT (USING EXISTING DATA)")
        print(f"{'='*50}")

        cmd = [
            sys.executable,
            "main_pipeline.py",
            "--config",
            config_path,
            "--use-existing-data",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=False)
            print(f"\n✅ {args.experiment.upper()} EXPERIMENT COMPLETED")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {args.experiment.upper()} EXPERIMENT FAILED: {e}")
    else:
        config = run_experiment(args.experiment, config_path)

        if config:
            print(f"\n📊 Results saved:")
            print(f"   - Dataset: {config['data_file']}")
            print(f"   - Results: {config['results_file']}")
            print(f"   - Report: {config['report_file']}")
            print(f"   - Plots: {config['plots_file']}")


if __name__ == "__main__":
    main()
