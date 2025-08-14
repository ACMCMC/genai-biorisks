"""
Extract CIDs from PubChem XML files for efficient compound collection
"""

import xml.etree.ElementTree as ET
import random
from typing import List, Set, Iterator
from tqdm import tqdm


class CIDExtractor:
    """Extract CIDs from PubChem XML files with SMILES length filtering"""

    def __init__(
        self,
        min_smiles_length: int,
        max_smiles_length: int,
        seed: int = 2262,
    ):
        self.min_smiles_length = min_smiles_length
        self.max_smiles_length = max_smiles_length
        self.cids = None
        random.seed(seed)

    def extract_all_cids(
        self, num_to_extract: int, pubchem_xml_file: str, t3db_xml: str
    ) -> List[int]:
        """Extract CIDs with valid SMILES length from the XML file, optionally merging with T3DB CIDs."""
        if self.cids is not None:
            return self.cids

        print(f"Extracting CIDs from {pubchem_xml_file}...")
        print(
            f"SMILES length filter: {self.min_smiles_length}-{self.max_smiles_length} characters"
        )

        cids = []

        # Optionally add T3DB CIDs first
        t3db_cids = set()
        if t3db_xml:
            t3db_cids = set(
                self.extract_cids_from_t3db(
                    t3db_xml,
                    max_cids=num_to_extract,
                    min_smiles_length=self.min_smiles_length,
                    max_smiles_length=self.max_smiles_length,
                )
            )
            print(f"Adding {len(t3db_cids)} T3DB CIDs to scan list.")
            cids.extend(t3db_cids)

        # Use iterparse for memory-efficient parsing of large XML
        try:
            context = ET.iterparse(pubchem_xml_file, events=("start", "end"))
            context = iter(context)
            event, root = next(context)

            compound_count = 0
            valid_compounds = 0

            for event, elem in tqdm(context, desc="Parsing XML"):
                if event == "end" and elem.tag.endswith("PC-Compound"):
                    compound_count += 1

                    # Extract both CID and SMILES from this compound
                    cid, smiles = self._extract_cid_and_smiles(elem)

                    if cid and smiles and self._is_valid_smiles_length(smiles):
                        # Avoid duplicates if T3DB CIDs already added
                        if cid not in t3db_cids:
                            cids.append(cid)
                        valid_compounds += 1

                        # Progress update every 1k valid compounds
                        if valid_compounds % 1000 == 0:
                            print(
                                f"Found {valid_compounds} valid CIDs (processed {compound_count} compounds)..."
                            )

                        # Stop if we've found enough valid CIDs
                        if (
                            num_to_extract is not None
                            and valid_compounds >= num_to_extract
                        ):
                            print(
                                f"Reached max_valid ({num_to_extract}) valid CIDs. Stopping early."
                            )
                            break

                    # Clear the element to free memory
                    elem.clear()

        except Exception as e:
            print(f"Error parsing XML: {e}")
            return []

        print(
            f"Successfully extracted {len(cids)} CIDs with valid SMILES length (including T3DB)"
        )
        print(
            f"Processed {compound_count} total compounds, kept {valid_compounds} ({(valid_compounds/compound_count*100) if compound_count else 0:.1f}%)"
        )
        self.cids = cids
        return cids

    def _extract_cid_and_smiles(self, compound_elem) -> tuple[int, str]:
        """Extract both CID and SMILES from a PC-Compound element"""

        cid = None
        smiles = None

        # Look for CID
        for elem in compound_elem.iter():
            if elem.tag.endswith("PC-CompoundType_id_cid"):
                try:
                    cid = int(elem.text)
                except (ValueError, TypeError):
                    pass
                break

        # Look for SMILES
        for info_data in compound_elem.iter():
            if info_data.tag.endswith("PC-InfoData"):
                # Check if this is a SMILES entry
                urn_label = None
                smiles_value = None

                for child in info_data.iter():
                    if child.tag.endswith("PC-Urn_label") and child.text == "SMILES":
                        urn_label = "SMILES"
                    elif child.tag.endswith("PC-InfoData_value_sval"):
                        smiles_value = child.text

                # If we found a SMILES entry, use the first one
                if urn_label == "SMILES" and smiles_value:
                    smiles = smiles_value
                    break

        return cid, smiles

    def _is_valid_smiles_length(self, smiles: str) -> bool:
        """Check if SMILES length is within valid range"""
        if not smiles:
            return False
        return self.min_smiles_length <= len(smiles) <= self.max_smiles_length

    def get_random_cids(self, count: int) -> List[int]:
        """Get random sample of CIDs"""

        if self.cids is None:
            self.extract_all_cids()

        if not self.cids:
            return []

        return random.sample(self.cids, min(count, len(self.cids)))

    def save_cids_to_file(self, output_file: str):
        """Save extracted CIDs to text file for future use"""

        if self.cids is None:
            self.extract_all_cids()

        print(f"Saving {len(self.cids)} CIDs to {output_file}...")

        with open(output_file, "w") as f:
            for cid in self.cids:
                f.write(f"{cid}\n")

        print(f"CIDs saved to {output_file}")

    @classmethod
    def load_cids_from_file(cls, cids_file: str) -> List[int]:
        """Load CIDs from text file"""

        print(f"Loading CIDs from {cids_file}...")

        cids = []
        with open(cids_file, "r") as f:
            for line in f:
                try:
                    cid = int(line.strip())
                    cids.append(cid)
                except ValueError:
                    continue

        print(f"Loaded {len(cids)} CIDs")
        return cids

    @staticmethod
    def extract_cids_from_t3db(
        toxins_xml_file: str, min_smiles_length, max_smiles_length, max_cids: int
    ) -> List[int]:
        """Extract PubChem CIDs from t3db/toxins.xml <compound> blocks, filtering by SMILES length."""
        import xml.etree.ElementTree as ET

        cids = set()
        block = []
        inside = False
        try:
            with open(toxins_xml_file, "r") as f:
                for line in f:
                    if "<compound" in line:
                        inside = True
                        block = [line]
                    elif inside:
                        block.append(line)
                        if "</compound>" in line:
                            inside = False
                            xml_str = "".join(block)
                            try:
                                elem = ET.fromstring(xml_str)
                                cid_elem = elem.find(".//pubchem_compound_id")
                                smiles_elem = elem.find(".//smiles")
                                if cid_elem is not None and smiles_elem is not None:
                                    smiles = smiles_elem.text.strip()
                                    if (
                                        smiles
                                        and min_smiles_length
                                        <= len(smiles)
                                        <= max_smiles_length
                                    ):
                                        cid = int(cid_elem.text.strip())
                                        cids.add(cid)
                                        if max_cids and len(cids) >= max_cids:
                                            break
                            except Exception:
                                continue
        except Exception as e:
            print(f"Error reading T3DB file: {e}")
        print(f"Extracted {len(cids)} CIDs from T3DB {toxins_xml_file}")
        return list(cids)
