import pandas as pd
import numpy as np
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings('ignore')
import os

def clean_smiles(smiles):
    """
    SMILES，Info
    
    SMILES|Info，：
    "OC1=CC2=C(C[C@@H]3[C@@H]4CCCC[C@]24CCN3CC=C)C=C1 |a:6,7,12,r,c:22,t:1,3,THB:16:15:3.4.5:7|"
    |Minute， reservedSMILESMinute
    """
    if pd.isna(smiles) or smiles == '':
        return None
    

    parts = str(smiles).split('|')
    cleaned_smiles = parts[0].strip()
    

    if cleaned_smiles == '':
        return None
    
    return cleaned_smiles

def smiles_to_morgan_fp(smiles, radius=2, n_bits=1024, use_chirality=True):
    """SMILESMorgan"""
    try:
        smiles=clean_smiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useChirality=use_chirality
        )
        
        fp_array = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        return fp_array
        
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None

def create_morgan_fingerprint_hdf5_h5py(data_dir,cell_type, hdf5_output_file, n_bits=1024, use_chirality=True):
    """
    h5pyMorganHDF5File，pytables
    """

    # print(f"Reading CSV file: {csv_file}")
    # df = pd.read_csv(csv_file)
    

    # required_columns = ['SAMPLE_KEY', 'SMILES']
    # for col in required_columns:
    #     if col not in df.columns:
    #         raise ValueError(f"CSV file must contain '{col}' column")


    split_index_name = cell_type
    files = ["train", "val", "test"]
    split_dir = data_dir

    dfs = []

    for file in files:
        split_path = os.path.join(split_dir, f"{split_index_name}-seen-{file}.csv")
        df = pd.read_csv(split_path)
        dfs.append(df)

    # Concatenate the DataFrames together
    df = pd.concat(dfs, ignore_index=False)
    

    fingerprints = []
    sample_keys = []
    failed_smiles = []
    
    print(f"Generating Morgan fingerprints for {len(df)} samples...")
    

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_key = row['SAMPLE_KEY']
        smiles = row['SMILES']
        

        fp = smiles_to_morgan_fp(smiles, n_bits=n_bits, use_chirality=use_chirality)
        
        if fp is not None:
            fingerprints.append(fp)

            sample_keys.append(str(sample_key))
        else:
            failed_smiles.append((sample_key, smiles))
    

    if failed_smiles:
        print(f"\nFailed to generate fingerprints for {len(failed_smiles)} SMILES:")
        for sample_key, smiles in failed_smiles[:10]:
            print(f"  {sample_key}: {smiles}")
        if len(failed_smiles) > 10:
            print(f"  ... and {len(failed_smiles) - 10} more")
    

    print(f"\nCreating array with {len(fingerprints)} valid fingerprints...")
    fingerprints_array = np.array(fingerprints, dtype=np.uint8)
    

    print(f"Saving to HDF5 using h5py: {hdf5_output_file}")
    
    with h5py.File(hdf5_output_file, 'w') as f:

        f.create_dataset("fingerprints", 
                        data=fingerprints_array, 
                        compression="gzip",
                        compression_opts=9)
        


        sample_keys_np = np.array(sample_keys, dtype=object)
        

        dt = h5py.special_dtype(vlen=str)
        sample_ds = f.create_dataset("sample_keys", 
                                     (len(sample_keys_np),), 
                                     dtype=dt,
                                     compression="gzip")
        
        for i, key in enumerate(sample_keys_np):
            sample_ds[i] = str(key)
        

        f.attrs['n_bits'] = n_bits
        f.attrs['use_chirality'] = use_chirality
        f.attrs['radius'] = 2
        f.attrs['description'] = "Morgan fingerprints (ECFP-like) for CellPainting compounds"
        f.attrs['created_by'] = "convert_smiles_to_morgan.py"
        f.attrs['n_samples'] = len(sample_keys)
    

    print("\nVerifying saved file...")
    with h5py.File(hdf5_output_file, 'r') as f:
        loaded_fingerprints = f["fingerprints"][:]
        loaded_keys = [str(key) for key in f["sample_keys"][:]]
        
        print(f"Loaded fingerprints shape: {loaded_fingerprints.shape}")
        print(f"Loaded keys shape: {len(loaded_keys)}")
        print(f"Number of keys: {len(loaded_keys)}")
        

        print("\nFirst 5 sample keys:")
        for i, key in enumerate(loaded_keys[:5]):
            print(f"  {i+1}. {key} (type: {type(key)})")
        

        print("\nFile attributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
    
    print(f"\n✅ Successfully created {hdf5_output_file}")
    print(f"   Total samples: {len(sample_keys)}")
    print(f"   Fingerprint length: {n_bits}")
    print(f"   Use chirality: {use_chirality}")
    
    return fingerprints_array, sample_keys

def verify_hdf5_structure(hdf5_file):
    """
    ValidationHDF5File
    """
    print(f"\nVerifying HDF5 structure of: {hdf5_file}")
    
    with h5py.File(hdf5_file, 'r') as f:
        print("HDF5 file structure:")
        
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        
        f.visititems(print_attrs)
    
    return True

def load_fingerprints_from_hdf5(hdf5_file):
    """
    HDF5FileLoad
    """
    with h5py.File(hdf5_file, 'r') as f:
        fingerprints = f["fingerprints"][:]

        sample_keys = []
        for key in f["sample_keys"][:]:
            if isinstance(key, bytes):
                sample_keys.append(key.decode('utf-8'))
            else:
                sample_keys.append(str(key))
    

    fp_dict = {}
    for key, fp in zip(sample_keys, fingerprints):
        fp_dict[str(key)] = fp
    
    return fp_dict

def create_pandas_compatible_hdf5(hdf5_file, output_file):
    """
    h5pyHDF5pandas
    """

    fp_dict = load_fingerprints_from_hdf5(hdf5_file)
    

    fp_df = pd.DataFrame.from_dict(fp_dict, orient='index')
    fp_df.index.name = 'SAMPLE_KEY'
    

    fp_df.to_hdf(output_file, key='df', mode='w')
    
    print(f"Created pandas-compatible HDF5: {output_file}")
    print(f"Shape: {fp_df.shape}")
    
    return fp_df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="datasets dir",
        required=True,
    )
    parser.add_argument("--hdf5_output", type=str, help="Types of contrastive pair.")
    parser.add_argument("--cell_type", type=str, help="Output file name.")

    return parser.parse_args()


if __name__ == "__main__":

    args=parse_args()
    data_dir = args.data_dir
    hdf5_output = args.hdf5_output
    cell_type=args.cell_type
    

    print("=" * 60)
    print("Generating Morgan Fingerprint HDF5 File (using h5py)")
    print("=" * 60)
    
    try:

        fp_array, sample_keys = create_morgan_fingerprint_hdf5_h5py(
            data_dir=data_dir,
            cell_type=cell_type,
            hdf5_output_file=hdf5_output,
            n_bits=1024,
            use_chirality=True
        )
        

        print("\n" + "=" * 60)
        print("Final Verification")
        print("=" * 60)
        verify_hdf5_structure(hdf5_output)
        

        print("\n" + "=" * 60)
        print("Test Loading")
        print("=" * 60)
        fp_dict = load_fingerprints_from_hdf5(hdf5_output)
        print(f"Loaded {len(fp_dict)} fingerprints")
        

        first_key = sample_keys[0] if sample_keys else None
        if first_key and first_key in fp_dict:
            print(f"\nFirst sample key: '{first_key}' (type: {type(first_key)})")
            print(f"Fingerprint shape: {fp_dict[first_key].shape}")
            print(f"Number of set bits: {fp_dict[first_key].sum()}")
        else:
            print(f"\nFirst sample key '{first_key}' not found in dictionary")
            print("Available keys (first 5):")
            for i, key in enumerate(list(fp_dict.keys())[:5]):
                print(f"  {i+1}. '{key}' (type: {type(key)})")
        

        # print("\n" + "=" * 60)
        # print("Creating pandas-compatible version (optional)")
        # print("=" * 60)
        
        # pandas_hdf5 = hdf5_output.replace(".hdf5", "_pandas.hdf5")
        # fp_df = create_pandas_compatible_hdf5(hdf5_output, pandas_hdf5)
        
        # print(f"\n✅ All files created successfully!")


        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()