"""
Efficient PubChem collector using pre-extracted CIDs
"""

import requests
import pandas as pd
import time
import random
import re
from typing import List, Dict, Optional
from tqdm import tqdm
from cid_extractor import CIDExtractor


class EfficientPubChemCollector:
    """Efficient collector using pre-extracted valid CIDs"""

    def __init__(
        self,
        pubchem_xml_file: str,
        t3db_xml_file: str,
        min_smiles_length: int,
        max_smiles_length: int,
        max_compound_cids_to_extract: int,
        delay: float = 0.1,
        seed: int = 2262,
    ):
        self.delay = delay
        random.seed(seed)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"

        self.min_smiles_length = min_smiles_length
        self.max_smiles_length = max_smiles_length

        extractor = CIDExtractor(
            min_smiles_length=min_smiles_length,
            max_smiles_length=max_smiles_length,
            seed=seed,
        )
        self.available_cids = extractor.extract_all_cids(
            num_to_extract=max_compound_cids_to_extract,
            pubchem_xml_file=pubchem_xml_file,
            t3db_xml=t3db_xml_file,
        )

        # # Randomly permute the self.available_cids
        # random.shuffle(self.available_cids)

        print(f"Loaded {len(self.available_cids)} valid CIDs for sampling")

    def get_compound_data(self, cid: int) -> Dict:
        """Get ALL compound data from a single API call (reusing existing method)"""

        try:
            url = f"{self.base_url}/{cid}/JSON/"
            response = requests.get(url, timeout=30)
            time.sleep(self.delay)

            if response.status_code != 200:
                return None

            data = response.json()

            # Extract ALL information from this single response
            compound_data = {
                "cid": cid,
                "name": self._extract_name(data),
                "smiles": self._extract_smiles(data),
                "molecular_formula": self._extract_molecular_formula(data),
                "molecular_weight": self._extract_molecular_weight(data),
                "iupac_name": self._extract_iupac_name(data),
                "h_codes": self._extract_h_codes(data),
                "ghs_pictograms": self._extract_ghs_pictograms(data),
                "hazard_statements": self._extract_hazard_statements(data),
            }

            # Determine overall toxicity
            compound_data["toxicity_type"] = self._classify_toxicity(compound_data)

            return compound_data

        except Exception as e:
            print(f"Warning: Error fetching data for CID {cid}: {e}")
            return None

    def collect_compounds_efficient(
        self,
        target_toxic_health: int,
        target_toxic_physical: int,
        target_nontoxic: int,
        max_attempts: int,
    ) -> pd.DataFrame:
        """Collect compounds using pre-extracted CIDs (much more efficient)"""

        print(
            f"Collecting compounds using {len(self.available_cids)} pre-extracted CIDs:"
        )
        print(
            f"Target: {target_toxic_health} toxic_health, {target_toxic_physical} toxic_physical, {target_nontoxic} nontoxic"
        )

        compounds = []
        toxic_health_count = 0
        toxic_physical_count = 0
        nontoxic_count = 0
        unknown_count = 0
        api_failures = 0
        attempts = 0

        pbar = tqdm(
            total=target_toxic_health + target_toxic_physical + target_nontoxic,
            desc="Collecting",
        )

        for cid in self.available_cids:
            if (
                toxic_health_count >= target_toxic_health
                and toxic_physical_count >= target_toxic_physical
                and nontoxic_count >= target_nontoxic
            ) or attempts >= max_attempts:
                break

            attempts += 1
            compound_data = self.get_compound_data(cid)

            if not compound_data:
                api_failures += 1
                continue

            # Must have a SMILES and have the correct SMILES length to be valid
            if not (
                self.min_smiles_length
                <= len(compound_data.get("smiles", ""))
                <= self.max_smiles_length
            ):
                continue

            toxicity_type = compound_data.get("toxicity_type", "unknown")

            # Consider both toxic_health and toxic_physical as "toxic" for collection
            if (
                toxicity_type == "toxic_health"
                and toxic_health_count < target_toxic_health
            ):
                compounds.append(compound_data)
                toxic_health_count += 1
                pbar.update(1)

            elif (
                toxicity_type == "toxic_physical"
                and toxic_physical_count < target_toxic_physical
            ):
                compounds.append(compound_data)
                toxic_physical_count += 1
                pbar.update(1)

            elif toxicity_type == "nontoxic" and nontoxic_count < target_nontoxic:
                compounds.append(compound_data)
                nontoxic_count += 1
                pbar.update(1)

            elif toxicity_type == "unknown":
                unknown_count += 1

            pbar.set_description(
                f"toxic_health: {toxic_health_count}, toxic_physical: {toxic_physical_count}, nontoxic: {nontoxic_count}, unknown: {unknown_count}"
            )

            # Progress every 100 attempts
            if attempts % 100 == 0:
                success_rate = len(compounds) / attempts * 100
                print(
                    f"\nAfter {attempts} attempts: Found={len(compounds)}, API failures={api_failures}, Unknown={unknown_count}, Success rate={success_rate:.1f}%"
                )

        pbar.close()

        success_rate = len(compounds) / attempts * 100 if attempts > 0 else 0
        print(f"\nCollection complete:")
        print(f"  Total compounds: {len(compounds)}")
        print(f"  Toxic: {toxic_health_count}, Non-toxic: {nontoxic_count}")
        print(f"  Attempts: {attempts}, Success rate: {success_rate:.1f}%")
        print(f"  API failures: {api_failures}, Unknown toxicity: {unknown_count}")

        if len(compounds) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(compounds)

        return df

    # All the extraction methods from SimplePubChemCollector (unchanged)
    def _extract_name(self, data: Dict) -> str:
        if "Record" in data and "RecordTitle" in data["Record"]:
            return data["Record"]["RecordTitle"]
        if "Record" in data and "Section" in data["Record"]:
            name = self._find_in_sections(data["Record"]["Section"], "name")
            if name:
                return name
        return f"Compound_{data.get('cid', 'Unknown')}"

    def _extract_smiles(self, data: Dict) -> str:
        return self._find_in_sections(
            data.get("Record", {}).get("Section", []), "smiles", ""
        )

    def _extract_molecular_formula(self, data: Dict) -> str:
        return self._find_in_sections(
            data.get("Record", {}).get("Section", []), "molecular formula", ""
        )

    def _extract_molecular_weight(self, data: Dict) -> float:
        weight_str = self._find_in_sections(
            data.get("Record", {}).get("Section", []), "molecular weight", ""
        )
        if weight_str:
            match = re.search(r"(\d+\.?\d*)", weight_str)
            if match:
                return float(match.group(1))
        return None

    def _extract_iupac_name(self, data: Dict) -> str:
        return self._find_in_sections(
            data.get("Record", {}).get("Section", []), "iupac name", ""
        )

    def _extract_h_codes(self, data: Dict) -> List[str]:
        h_codes = []
        if "Record" in data and "Section" in data["Record"]:
            h_codes = self._find_h_codes_in_sections(data["Record"]["Section"])
        return list(set(h_codes))

    def _extract_ghs_pictograms(self, data: Dict) -> List[str]:
        pictograms = []
        if "Record" in data and "Section" in data["Record"]:
            pictograms = self._find_ghs_pictograms_in_sections(
                data["Record"]["Section"]
            )
        return list(set(pictograms))

    def _extract_hazard_statements(self, data: Dict) -> List[str]:
        statements = []
        if "Record" in data and "Section" in data["Record"]:
            statements = self._find_hazard_statements_in_sections(
                data["Record"]["Section"]
            )
        return statements

    def _find_in_sections(self, sections: List, search_term: str, default=None):
        for section in sections:
            heading = section.get("TOCHeading", "").lower()
            if search_term.lower() in heading:
                value = self._extract_value_from_section(section)
                if value:
                    return value
            if "Information" in section:
                for info in section["Information"]:
                    if "Name" in info and search_term.lower() in info["Name"].lower():
                        value = self._extract_value_from_info(info)
                        if value:
                            return value
            if "Section" in section:
                result = self._find_in_sections(
                    section["Section"], search_term, default
                )
                if result != default:
                    return result
        return default

    def _find_h_codes_in_sections(self, sections: List) -> List[str]:
        h_codes = []
        for section in sections:
            heading = section.get("TOCHeading", "").lower()
            if any(term in heading for term in ["ghs", "hazard", "safety"]):
                h_codes.extend(self._extract_h_codes_from_section(section))
            if "Section" in section:
                h_codes.extend(self._find_h_codes_in_sections(section["Section"]))
        return h_codes

    def _find_ghs_pictograms_in_sections(self, sections: List) -> List[str]:
        pictograms = []
        for section in sections:
            heading = section.get("TOCHeading", "").lower()
            if any(term in heading for term in ["primary hazard", "ghs", "pictogram"]):
                pictograms.extend(self._extract_pictograms_from_section(section))
            if "Section" in section:
                pictograms.extend(
                    self._find_ghs_pictograms_in_sections(section["Section"])
                )
        return pictograms

    def _find_hazard_statements_in_sections(self, sections: List) -> List[str]:
        statements = []
        for section in sections:
            heading = section.get("TOCHeading", "").lower()
            if "hazard statement" in heading:
                statements.extend(self._extract_statements_from_section(section))
            if "Section" in section:
                statements.extend(
                    self._find_hazard_statements_in_sections(section["Section"])
                )
        return statements

    def _extract_value_from_section(self, section: Dict) -> str:
        if "Information" in section:
            for info in section["Information"]:
                value = self._extract_value_from_info(info)
                if value:
                    return value
        return None

    def _extract_value_from_info(self, info: Dict) -> str:
        if "Value" in info and "StringWithMarkup" in info["Value"]:
            for string_item in info["Value"]["StringWithMarkup"]:
                if "String" in string_item and string_item["String"].strip():
                    return string_item["String"].strip()
        return None

    def _extract_h_codes_from_section(self, section: Dict) -> List[str]:
        h_codes = []
        if "Information" in section:
            for info in section["Information"]:
                if "Value" in info and "StringWithMarkup" in info["Value"]:
                    for string_item in info["Value"]["StringWithMarkup"]:
                        if "String" in string_item:
                            text = string_item["String"]
                            codes = re.findall(r"H\d{3}", text)
                            h_codes.extend(codes)
        if "Section" in section:
            for subsection in section["Section"]:
                h_codes.extend(self._extract_h_codes_from_section(subsection))
        return h_codes

    def _extract_pictograms_from_section(self, section: Dict) -> List[str]:
        pictograms = []
        if "Information" in section:
            for info in section["Information"]:
                if "Value" in info and "StringWithMarkup" in info["Value"]:
                    for string_item in info["Value"]["StringWithMarkup"]:
                        if "Markup" in string_item:
                            for markup in string_item["Markup"]:
                                if "URL" in markup and "GHS" in markup["URL"]:
                                    match = re.search(r"GHS(\d+)\.svg", markup["URL"])
                                    if match:
                                        ghs_num = match.group(1)
                                        pictogram = self._ghs_number_to_name(ghs_num)
                                        if pictogram:
                                            pictograms.append(pictogram)
                                if "Extra" in markup:
                                    extra = markup["Extra"]
                                    if extra in [
                                        "Flammable",
                                        "Health Hazard",
                                        "Irritant",
                                        "Environmental Hazard",
                                    ]:
                                        pictograms.append(extra)
        return pictograms

    def _extract_statements_from_section(self, section: Dict) -> List[str]:
        statements = []
        if "Information" in section:
            for info in section["Information"]:
                statement = self._extract_value_from_info(info)
                if statement and len(statement) > 10:
                    statements.append(statement)
        return statements

    def _ghs_number_to_name(self, ghs_number: str) -> str:
        ghs_map = {
            "01": "Explosives",
            "02": "Flammables",
            "03": "Oxidizers",
            "04": "Compressed Gases",
            "05": "Corrosives",
            "06": "Acute Toxicity",
            "07": "Irritant",
            "08": "Health Hazard",
            "09": "Environment",
        }
        return ghs_map.get(ghs_number, f"GHS{ghs_number}")

    def _classify_toxicity(self, compound_data: Dict) -> str:
        """Classify toxicity based on GHS pictograms"""
        h_codes = compound_data.get("h_codes", [])
        pictograms = compound_data.get("ghs_pictograms", [])

        # toxic_health: Has GHS08 (Health Hazard pictogram)
        if "Health Hazard" in pictograms:
            return "toxic_health"

        # toxic_physical: Has GHS codes but not GHS08
        if h_codes and "Health Hazard" not in pictograms:
            return "toxic_physical"

        # nontoxic: No GHS codes at all
        if not h_codes:
            return "nontoxic"

        return "unknown"
