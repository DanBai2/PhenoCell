import pandas as pd
import numpy as np
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def smiles_to_morgan_fp(smiles, radius=2, n_bits=1024, use_chirality=True):
    """
    SMILESMorgan
    
    Parameter:
    - smiles: SMILES
    - radius: Morgan（2ECFP4）
    - n_bits: （1024）
    - use_chirality: 
    
    Returns:
    - morgan_fp: n_bitsnumpy
    """
    try:

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        

        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=radius, 
            nBits=n_bits,
            useChirality=use_chirality
        )
        

        fp_array = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(morgan_fp, fp_array)
        return fp_array
        
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

def create_morgan_fingerprint_hdf5(csv_file, hdf5_output_file, n_bits=1024, use_chirality=True):
    """
    CSVFileMorganHDF5File
    
    Parameter:
    - csv_file: SMILESCSVFilePATH
    - hdf5_output_file: HDF5FilePATH
    - n_bits: 
    - use_chirality: 
    """

    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    

    required_columns = ['SAMPLE_KEY', 'SMILES']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain '{col}' column")
    

    fingerprints = {}
    sample_keys = []
    failed_smiles = []
    
    print(f"Generating Morgan fingerprints for {len(df)} samples...")
    

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_key = row['SAMPLE_KEY']
        smiles = row['SMILES']
        

        fp = smiles_to_morgan_fp(smiles, n_bits=n_bits, use_chirality=use_chirality)
        
        if fp is not None:
            fingerprints[sample_key] = fp
            sample_keys.append(sample_key)
        else:
            failed_smiles.append((sample_key, smiles))
    

    if failed_smiles:
        print(f"\nFailed to generate fingerprints for {len(failed_smiles)} SMILES:")
        for sample_key, smiles in failed_smiles[:10]:
            print(f"  {sample_key}: {smiles}")
        if len(failed_smiles) > 10:
            print(f"  ... and {len(failed_smiles) - 10} more")
    

    print(f"\nCreating DataFrame with {len(fingerprints)} valid fingerprints...")
    

    fp_df = pd.DataFrame.from_dict(fingerprints, orient='index')
    fp_df.index.name = 'SAMPLE_KEY'
    

    fp_df.columns = list(range(n_bits))
    

    print(f"Saving to HDF5: {hdf5_output_file}")
    fp_df.to_hdf(hdf5_output_file, key='df', mode='w')
    

    print("\nVerifying saved file...")
    with pd.HDFStore(hdf5_output_file, 'r') as store:
        loaded_df = store['df']
        print(f"Loaded shape: {loaded_df.shape}")
        print(f"Loaded index name: {loaded_df.index.name}")
        print("\nFirst few entries:")
        print(loaded_df.head())
    
    print(f"\n✅ Successfully created {hdf5_output_file}")
    print(f"   Total samples: {len(loaded_df)}")
    print(f"   Fingerprint length: {n_bits}")
    print(f"   Use chirality: {use_chirality}")
    
    return fp_df

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
    

    df = pd.read_hdf(hdf5_file, key='df')
    print(f"\nDataFrame info:")
    print(f"  Shape: {df.shape}")
    print(f"  Index name: {df.index.name}")
    print(f"  First index value: {df.index[0]}")
    print(f"  Fingerprint length: {df.shape[1]}")
    
    return df


if __name__ == "__main__":

    csv_file = "path/to/your/project/models/cellpainting-split-test-imgpermol.csv"
    hdf5_output = "path/to/your/data/bray2017/metadata/morgan_chiral_fps_1024.hdf5"
    

    print("=" * 60)
    print("Generating Morgan Fingerprint HDF5 File")
    print("=" * 60)
    
    try:

        fp_df = create_morgan_fingerprint_hdf5(
            csv_file=csv_file,
            hdf5_output_file=hdf5_output,
            n_bits=1024,
            use_chirality=True
        )
        

        print("\n" + "=" * 60)
        print("Final Verification")
        print("=" * 60)
        verify_hdf5_structure(hdf5_output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()