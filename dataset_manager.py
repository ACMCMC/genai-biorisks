"""
Dataset management using Hugging Face datasets for reproducibility
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Optional, Dict, Any
from efficient_pubchem_collector import EfficientPubChemCollector


class DatasetManager:
    """Manages compound datasets with local storage and versioning"""

    def __init__(self, cache_dir: str = "datasets_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def create_and_cache_dataset(
        self,
        name: str,
        total_compounds: int,
        min_smiles_length: int,
        max_smiles_length: int,
        pubchem_xml_file: str,
        t3db_xml_file: str,
        version: str = "v1.0",
        force_recreate: bool = False,
    ) -> Dataset:
        """Create a new dataset and cache it locally"""

        dataset_path = os.path.join(self.cache_dir, f"{name}_{version}")

        # Check if dataset already exists
        if os.path.exists(dataset_path) and not force_recreate:
            print(f"Loading existing dataset from {dataset_path}")
            return Dataset.load_from_disk(dataset_path)

        print(f"Creating new dataset '{name}' with {total_compounds} compounds...")
        print(
            f"SMILES length filter: {min_smiles_length}-{max_smiles_length} characters"
        )

        # Create dataset using compound collector with integrated CID extraction
        collector = EfficientPubChemCollector(
            pubchem_xml_file=pubchem_xml_file,
            t3db_xml_file=t3db_xml_file,
            delay=0.1,
            seed=2262,
            min_smiles_length=min_smiles_length,
            max_smiles_length=max_smiles_length,
            max_compound_cids_to_extract=total_compounds * 20,
        )
        compounds_df = collector.collect_compounds_efficient(
            target_toxic_health=total_compounds // 2,
            target_toxic_physical=0,
            target_nontoxic=total_compounds // 2,
            max_attempts=total_compounds * 20,
        )

        # Assert we have the expected counts of each type
        assert (
            compounds_df["toxicity_type"].value_counts().get("toxic_health", 0)
            == total_compounds // 2
        )
        assert (
            compounds_df["toxicity_type"].value_counts().get("nontoxic", 0)
            == total_compounds // 2
        )

        # Clean up data types for Arrow compatibility
        compounds_df = self._clean_dataframe_for_arrow(compounds_df)

        # Convert to Hugging Face dataset
        dataset = Dataset.from_pandas(compounds_df)

        # Add metadata
        dataset = dataset.add_column("dataset_name", [name] * len(dataset))
        dataset = dataset.add_column("dataset_version", [version] * len(dataset))
        dataset = dataset.add_column(
            "creation_date", [pd.Timestamp.now().isoformat()] * len(dataset)
        )

        # Save to disk
        dataset.save_to_disk(dataset_path)
        print(f"Dataset saved to {dataset_path}")

        # Also save as CSV for compatibility
        csv_path = f"{dataset_path}.csv"
        compounds_df.to_csv(csv_path, index=False)
        print(f"CSV version saved to {csv_path}")

        return dataset

    def load_dataset(self, name: str, version: str = "v1.0") -> Optional[Dataset]:
        """Load a cached dataset"""

        dataset_path = os.path.join(self.cache_dir, f"{name}_{version}")

        if not os.path.exists(dataset_path):
            print(f"Dataset {name}_{version} not found in {self.cache_dir}")
            return None

        print(f"Loading dataset from {dataset_path}")
        return Dataset.load_from_disk(dataset_path)

    def _clean_dataframe_for_arrow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame for Arrow/Datasets compatibility"""
        df = df.copy()

        # Handle mixed types and missing values
        for col in df.columns:
            if col == "molecular_weight":
                # Convert to float, handle None/NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif col == "categories":
                # Convert list to string representation
                df[col] = df[col].astype(str)
            else:
                # Convert other object columns to string, handle None
                if df[col].dtype == "object":
                    df[col] = df[col].fillna("").astype(str)

        return df


def load_dataset_for_experiment(
    total_compounds: int,
    min_smiles_length: int,
    max_smiles_length: int,
    pubchem_xml_file: str,
    t3db_xml_file: str,
) -> pd.DataFrame:
    """Load dataset for experiments, converting to pandas DataFrame"""

    manager = DatasetManager()

    dataset_name = f"compounds_{total_compounds}"
    dataset = manager.load_dataset(dataset_name, version="v1.0")

    if dataset is None:
        print(f"Dataset {dataset_name} not found. Creating it...")
        dataset = manager.create_and_cache_dataset(
            name=dataset_name,
            total_compounds=total_compounds,
            min_smiles_length=min_smiles_length,
            max_smiles_length=max_smiles_length,
            version="v1.0",
            pubchem_xml_file=pubchem_xml_file,
            t3db_xml_file=t3db_xml_file,
        )

    # Convert to pandas for compatibility with existing pipeline
    return dataset.to_pandas()
