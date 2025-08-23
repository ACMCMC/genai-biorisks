"""
Main pipeline for language model toxicity assessment
"""

import argparse
import os
import pandas as pd
import yaml
from efficient_pubchem_collector import EfficientPubChemCollector
from model_evaluation import ToxicityEvaluator
from statistical_analysis import ToxicityAnalyzer
from dataset_manager import DatasetManager, load_dataset_for_experiment
import gc
import torch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Language Model Toxicity Assessment Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--use-existing-data",
        action="store_true",
        help="Use existing dataset file instead of collecting new data",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract config values
    compounds = config["compounds"]
    models = config["models"]
    data_file = config["data_file"]
    results_file = config["results_file"]
    report_file = config["report_file"]
    plots_file = config["plots_file"]
    min_smiles_length = config["min_smiles_length"]
    max_smiles_length = config["max_smiles_length"]
    pubchem_xml_file = config["pubchem_xml_file"]
    t3db_xml_file = config["t3db_xml_file"]

    # Step 1: Data Collection
    print(f"=== STEP 1: LOADING CACHED DATASET ===")
    compounds_df = load_dataset_for_experiment(
        total_compounds=compounds,
        min_smiles_length=min_smiles_length,
        max_smiles_length=max_smiles_length,
        pubchem_xml_file=pubchem_xml_file,
        t3db_xml_file=t3db_xml_file,
    )
    compounds_df.to_csv(data_file, index=False)

    print(f"Dataset loaded: {len(compounds_df)} compounds")

    # Step 2: Model Evaluation
    print("\n=== STEP 2: MODEL EVALUATION ===")
    all_results = []

    for model_name in models:
        print(f"\nEvaluating model: {model_name}")

        evaluator = None

        try:
            # Clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()
            # Assert that at least 95% of GPU memory is free
            assert (
                torch.cuda.memory_reserved()
                / torch.cuda.get_device_properties(0).total_memory
                < 0.95
            ), "GPU memory not sufficiently freed"
            evaluator = ToxicityEvaluator(model_name)
            results = evaluator.evaluate_compounds(compounds_df, char_cutoff=min_smiles_length)
            all_results.append(results)
            print(f"Completed evaluation of {len(results)} compounds")

        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(results_file, index=False)
        print(f"\nEvaluation results saved to {results_file}")

        # Step 3: Statistical Analysis
        print("\n=== STEP 3: STATISTICAL ANALYSIS ===")
        analyzer = ToxicityAnalyzer(combined_results)

        # Generate report (automatically handles single vs multi-model)
        report = analyzer.generate_summary_report()
        print(report)

        # Save report
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Report saved to {report_file}")

        # Create visualizations (automatically handles single vs multi-model)
        analyzer.create_visualizations(plots_file)

        # Per-model summary statistics
        print("\n=== PER-MODEL SUMMARY ===")
        models = combined_results["model"].unique()

        any_significant = False

        for model_name in models:
            model_data = combined_results[combined_results["model"] == model_name]
            model_analyzer = ToxicityAnalyzer(model_data)

            perp_results = model_analyzer.perform_statistical_tests("perplexity")
            rank_results = model_analyzer.perform_statistical_tests("mean_rank")

            perp_sig = perp_results.get("mannwhitney_significant", False)
            rank_sig = rank_results.get("mannwhitney_significant", False)

            if perp_sig or rank_sig:
                any_significant = True
                print(f"\n⚠️  {model_name} - SIGNIFICANT DIFFERENCES DETECTED")
                if perp_sig:
                    print(
                        f"   Perplexity: Cliff's Delta = {perp_results['cliff_delta']:.4f} ({perp_results['cliff_delta_interpretation']})"
                    )
                if rank_sig:
                    print(
                        f"   Mean Rank: Cliff's Delta = {rank_results['cliff_delta']:.4f} ({rank_results['cliff_delta_interpretation']})"
                    )
            else:
                print(f"\n✅ {model_name} - NO SIGNIFICANT DIFFERENCES DETECTED")

        if not any_significant:
            print(
                f"\n✅ OVERALL: NO SIGNIFICANT DIFFERENCES DETECTED ACROSS {len(models)} MODEL(S)"
            )
        else:
            print(f"\n⚠️  OVERALL: POTENTIAL BIOSECURITY RISKS DETECTED IN SOME MODELS")

    else:
        print("❌ No models were successfully evaluated")

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
