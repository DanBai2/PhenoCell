"""
Convert CAS numbers to SMILES via PubChem and generate Morgan fingerprints.

Usage:
    python scripts/preprocessing/convert_cas_to_smiles.py \
        --annotation path/to/annotation.xlsx \
        --output_dir path/to/output \
        --fingerprint_bits 1024 \
        --request_delay 0.3
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pubchempy as pcp
except ImportError:
    pcp = None
    print("Warning: pubchempy not installed. Run: pip install pubchempy")
    print("  Script will skip SMILES lookup and produce empty columns only.")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not installed. Run: pip install rdkit")
    print("  Script will skip fingerprint generation.")


def query_smiles_pubchem(cas, name):
    """Query PubChem for canonical SMILES by CAS number, falling back to compound name."""
    if pd.isna(cas) or not str(cas).strip():
        tqdm.write(f"  No CAS provided, skipping -> Name: {name}")
        return None
    if pcp is None:
        return None

    cas_clean = str(cas).strip()
    try:
        compounds = pcp.get_compounds(cas_clean, "name")
        if compounds and compounds[0].canonical_smiles:
            return compounds[0].canonical_smiles
    except Exception:
        pass

    # Fallback: search by compound name
    name_clean = str(name).strip() if pd.notna(name) else ""
    if name_clean and name_clean != "nan":
        try:
            compounds = pcp.get_compounds(name_clean, "name")
            if compounds and compounds[0].canonical_smiles:
                return compounds[0].canonical_smiles
        except Exception:
            pass

    return None


def smiles_to_fingerprint(smiles, n_bits=1024, radius=2):
    """Convert SMILES string to Morgan fingerprint bit vector."""
    if not HAS_RDKIT or not smiles or pd.isna(smiles):
        return np.zeros(n_bits, dtype=np.float32)
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CAS numbers to SMILES via PubChem and generate Morgan fingerprints."
    )
    parser.add_argument(
        "--annotation",
        type=str,
        required=True,
        help="Path to input annotation file (.xlsx or .csv). Must contain 'cas' and 'name' columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--output_smiles",
        type=str,
        default="annotation_with_smiles.xlsx",
        help="Filename for the output annotation with SMILES column (default: annotation_with_smiles.xlsx)",
    )
    parser.add_argument(
        "--output_fingerprints",
        type=str,
        default="drug_fingerprints.npz",
        help="Filename for the fingerprint dictionary .npz (default: drug_fingerprints.npz)",
    )
    parser.add_argument(
        "--output_matrix",
        type=str,
        default="drug_fingerprints_matrix.npz",
        help="Filename for the fingerprint matrix .npz (default: drug_fingerprints_matrix.npz)",
    )
    parser.add_argument(
        "--fingerprint_bits",
        type=int,
        default=1024,
        help="Number of bits for Morgan fingerprints (default: 1024)",
    )
    parser.add_argument(
        "--fingerprint_radius",
        type=int,
        default=2,
        help="Radius for Morgan fingerprints (default: 2)",
    )
    parser.add_argument(
        "--request_delay",
        type=float,
        default=0.3,
        help="Delay in seconds between PubChem requests to avoid rate limiting (default: 0.3)",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="index",
        help="Column name to use as drug/sample ID for fingerprint keys (default: index)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load annotation table
    anno_path = args.annotation
    if anno_path.endswith(".csv"):
        anno = pd.read_csv(anno_path)
    else:
        anno = pd.read_excel(anno_path)
    print(f"Loaded annotation table: {len(anno)} rows, columns: {list(anno.columns)}")

    # 2. Query SMILES via PubChem
    smiles_list = []
    n_found = 0
    for idx, row in tqdm(anno.iterrows(), total=len(anno), desc="Querying SMILES"):
        cas = row.get("cas", None)
        name = row.get("name", str(idx))
        smi = query_smiles_pubchem(cas, name)
        smiles_list.append(smi)
        if smi:
            n_found += 1
        if pcp and idx % 20 == 0:
            time.sleep(args.request_delay)

    anno["SMILES"] = smiles_list
    print(f"SMILES query complete: {n_found}/{len(anno)} ({n_found / len(anno) * 100:.1f}%)")

    # 3. Save annotation with SMILES
    out_xlsx = os.path.join(args.output_dir, args.output_smiles)
    anno.to_excel(out_xlsx, index=False)
    print(f"Annotation saved to: {out_xlsx}")

    # 4. Generate Morgan fingerprints
    if HAS_RDKIT:
        fingerprints = {}
        n_fp = 0
        for _, row in tqdm(anno.iterrows(), total=len(anno), desc="Generating fingerprints"):
            cid = str(row.get(args.id_column, ""))
            smi = row.get("SMILES", None)
            fp = smiles_to_fingerprint(smi, n_bits=args.fingerprint_bits, radius=args.fingerprint_radius)
            if fp.sum() > 0:
                n_fp += 1
            fingerprints[cid] = fp

        # Save as drug_id -> fingerprint dict
        fp_path = os.path.join(args.output_dir, args.output_fingerprints)
        np.savez_compressed(fp_path, **fingerprints)
        print(f"Fingerprints saved: {fp_path} ({n_fp}/{len(anno)} valid)")

        # Save as matrix format (sorted by drug_id)
        try:
            all_ids = sorted(fingerprints.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        except (ValueError, TypeError):
            all_ids = sorted(fingerprints.keys())
        fp_matrix = np.stack([fingerprints[cid] for cid in all_ids])
        matrix_path = os.path.join(args.output_dir, args.output_matrix)
        np.savez_compressed(matrix_path, fingerprints=fp_matrix, drug_ids=np.array(all_ids))
        print(f"Fingerprint matrix saved: {matrix_path} ({len(all_ids)} drugs x {args.fingerprint_bits} bits)")

    # 5. Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total drugs:       {len(anno)}")
    print(f"  SMILES found:      {n_found}")
    if HAS_RDKIT:
        print(f"  Valid fingerprints:{n_fp}")
    print(f"  Output directory:  {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
