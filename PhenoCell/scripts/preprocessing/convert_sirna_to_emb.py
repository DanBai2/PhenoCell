import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings('ignore')

class ProjectionLayer(nn.Module):
    """：1024"""
    def __init__(self, input_dim=768, output_dim=1024):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.projection(x)
        x = self.layer_norm(x)
        return x

def sirna_to_bert_embedding(sirna_id, tokenizer, model, projection_layer, device, max_length=128):
    """siRNA IDBERTEmbedding，1024"""
    try:

        text = f"siRNA identifier: {sirna_id}"
        

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        

        inputs = {k: v.to(device) for k, v in inputs.items()}
        

        with torch.no_grad():
            outputs = model(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
            

            if projection_layer is not None:
                cls_embedding = projection_layer(cls_embedding)
            
            cls_embedding = cls_embedding.cpu().numpy()
        
        return cls_embedding[0]
        
    except Exception as e:
        print(f"Error processing siRNA {sirna_id}: {e}")
        return None

def load_bert_model(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", 
                   projection_dim=1024):
    """LoadPubMedBERT、Minute"""
    print(f"Loading PubMedBERT model: {model_name}")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    

    # with torch.no_grad():
    #     dummy_input = tokenizer("test", return_tensors="pt").to(device)
    #     dummy_output = model(dummy_input)
    #     model_dim = dummy_output.last_hidden_state.shape[-1]
    # print(f"Original model dimension: {model_dim}")
    model_dim=768
    

    if model_dim != projection_dim:
        projection_layer = ProjectionLayer(input_dim=768, output_dim=projection_dim)
        projection_layer = projection_layer.to(device)
        projection_layer.eval()
        print(f"Created projection layer: {model_dim} -> {projection_dim}")
    else:
        projection_layer = None
        print(f"No projection needed, model dimension is already {projection_dim}")
    

    model = model.to(device)
    model.eval()
    
    return tokenizer, model, projection_layer, device

def create_rxrx1_bert_embeddings(csv_file, hdf5_output_file, 
                               model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                               batch_size=32, max_length=128,
                               target_dim=1024):
    """
    RXRX1DatasetBERTEmbeddingSaveHDF5File
    
    Parameters:
    -----------
    csv_file : str
        RXRX1 metadata.csvFilePATH
    hdf5_output_file : str
        HDF5FilePATH
    model_name : str
        BERTName
    batch_size : int
        ProcessingSize
    max_length : int
        
    target_dim : int
        Embedding（1024）
    """

    print("Loading BERT model and projection layer...")
    tokenizer, model, projection_layer, device = load_bert_model(model_name, projection_dim=target_dim)
    
    print(f"Target embedding dimension: {target_dim}")
    

    print(f"\nReading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    

    required_columns = ['site_id', 'sirna']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain '{col}' column")
    
    print(f"Found {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    

    print(f"\nUnique site_id count: {df['site_id'].nunique()}")
    print(f"Unique sirna count: {df['sirna'].nunique()}")
    print(f"First few samples:")
    print(df[['site_id', 'sirna']].head())
    

    all_embeddings = []
    all_site_ids = []
    all_sirnas = []
    failed_samples = []
    
    print(f"\nGenerating BERT embeddings for {len(df)} samples...")
    

    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        
        batch_embeddings = []
        batch_site_ids = []
        batch_sirnas = []
        
        for _, row in batch_df.iterrows():
            site_id = row['site_id']
            sirna = row['sirna']
            
            try:

                embedding = sirna_to_bert_embedding(sirna, tokenizer, model, projection_layer, device, max_length)
                
                if embedding is not None:
                    batch_embeddings.append(embedding)
                    batch_site_ids.append(str(site_id))
                    batch_sirnas.append(str(sirna))
                else:
                    failed_samples.append((site_id, sirna))
                    
            except Exception as e:
                print(f"Error processing {site_id}, {sirna}: {e}")
                failed_samples.append((site_id, sirna))
        
        if batch_embeddings:
            all_embeddings.extend(batch_embeddings)
            all_site_ids.extend(batch_site_ids)
            all_sirnas.extend(batch_sirnas)
    

    if failed_samples:
        print(f"\nFailed to generate embeddings for {len(failed_samples)} samples:")
        for site_id, sirna in failed_samples[:10]:
            print(f"  site_id: {site_id}, sirna: {sirna}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more")
    

    print(f"\nCreating arrays with {len(all_embeddings)} valid embeddings...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Embeddings dtype: {embeddings_array.dtype}")
    

    if embeddings_array.shape[1] != target_dim:
        print(f"Warning: Embedding dimension is {embeddings_array.shape[1]}, expected {target_dim}")
    else:
        print(f"✅ Successfully created {embeddings_array.shape[0]} embeddings with dimension {target_dim}")
    

    print(f"\nSaving to HDF5: {hdf5_output_file}")
    
    with h5py.File(hdf5_output_file, 'w') as f:

        f.create_dataset("embeddings", 
                        data=embeddings_array, 
                        compression="gzip",
                        compression_opts=9)
        

        site_ids_np = np.array(all_site_ids, dtype=object)
        dt = h5py.special_dtype(vlen=str)
        site_ds = f.create_dataset("site_ids", 
                                   (len(site_ids_np),), 
                                   dtype=dt,
                                   compression="gzip")
        
        for i, site_id in enumerate(site_ids_np):
            site_ds[i] = str(site_id)
        

        sirnas_np = np.array(all_sirnas, dtype=object)
        sirna_ds = f.create_dataset("sirnas", 
                                    (len(sirnas_np),), 
                                    dtype=dt,
                                    compression="gzip")
        
        for i, sirna in enumerate(sirnas_np):
            sirna_ds[i] = str(sirna)
        

        f.attrs['model_name'] = model_name
        f.attrs['embedding_dim'] = embeddings_array.shape[1]
        f.attrs['target_dim'] = target_dim
        f.attrs['max_length'] = max_length
        f.attrs['description'] = f"PubMedBERT embeddings for RXRX1 siRNA data (projected to {target_dim}D)"
        f.attrs['created_by'] = "create_rxrx1_embeddings.py"
        f.attrs['n_samples'] = len(all_site_ids)
        f.attrs['dataset'] = "RXRX1"
        f.attrs['original_csv'] = csv_file
        f.attrs['has_projection'] = projection_layer is not None
    

    print("\nVerifying saved file...")
    with h5py.File(hdf5_output_file, 'r') as f:
        loaded_embeddings = f["embeddings"][:]
        loaded_site_ids = [str(key) for key in f["site_ids"][:]]
        loaded_sirnas = [str(key) for key in f["sirnas"][:]]
        
        print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
        print(f"Loaded site_ids count: {len(loaded_site_ids)}")
        print(f"Loaded sirnas count: {len(loaded_sirnas)}")
        

        print("\nFirst 5 samples:")
        for i in range(min(5, len(loaded_site_ids))):
            print(f"  {i+1}. site_id: {loaded_site_ids[i]}, sirna: {loaded_sirnas[i]}")
        

        print("\nFile attributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
    
    print(f"\n✅ Successfully created {hdf5_output_file}")
    print(f"   Total samples: {len(all_site_ids)}")
    print(f"   Embedding dimension: {embeddings_array.shape[1]}")
    print(f"   Model: {model_name}")
    
    return embeddings_array, all_site_ids, all_sirnas

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

def load_embeddings_from_hdf5(hdf5_file):
    """
    HDF5FileLoadEmbedding
    """
    with h5py.File(hdf5_file, 'r') as f:
        embeddings = f["embeddings"][:]
        

        site_ids = []
        for key in f["site_ids"][:]:
            if isinstance(key, bytes):
                site_ids.append(key.decode('utf-8'))
            else:
                site_ids.append(str(key))
        

        sirnas = []
        for key in f["sirnas"][:]:
            if isinstance(key, bytes):
                sirnas.append(key.decode('utf-8'))
            else:
                sirnas.append(str(key))
    

    embedding_dict = {}
    for site_id, sirna, emb in zip(site_ids, sirnas, embeddings):
        embedding_dict[str(site_id)] = {
            'embedding': emb,
            'sirna': sirna
        }
    
    return embedding_dict

def create_simple_dataset_class(csv_file, hdf5_file, sample_size=None):
    """
    Dataset，TestLoad
    """
    print(f"\nCreating test dataset from {csv_file} and {hdf5_file}")
    

    df = pd.read_csv(csv_file)
    if sample_size:
        df = df.head(sample_size)
    

    embedding_dict = load_embeddings_from_hdf5(hdf5_file)
    

    class SimpleDataset:
        def __init__(self, df, embedding_dict):
            self.df = df
            self.embedding_dict = embedding_dict
            

            self.valid_indices = []
            for idx, row in df.iterrows():
                site_id = str(row['site_id'])
                if site_id in embedding_dict:
                    self.valid_indices.append(idx)
            
            print(f"Valid samples: {len(self.valid_indices)}/{len(df)}")
        
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            df_idx = self.valid_indices[idx]
            row = self.df.iloc[df_idx]
            site_id = str(row['site_id'])
            
            data = self.embedding_dict[site_id]
            
            return {
                'site_id': site_id,
                'sirna': data['sirna'],
                'embedding': data['embedding']
            }
    
    return SimpleDataset(df, embedding_dict)


if __name__ == "__main__":

    csv_file = "path/to/your/data/RXRX1/rxrx1/metadata.csv"
    hdf5_output = "path/to/your/data/RXRX1/sirna_emb/sirna_pubmedbert_embeddings_1024d.h5"
    
    print("=" * 60)
    print("Generating PubMedBERT Embeddings for RXRX1 Dataset")
    print("=" * 60)
    
    try:

        embeddings, site_ids, sirnas = create_rxrx1_bert_embeddings(
            csv_file=csv_file,
            hdf5_output_file=hdf5_output,
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            batch_size=16,
            max_length=64
        )
        

        print("\n" + "=" * 60)
        print("Final Verification")
        print("=" * 60)
        verify_hdf5_structure(hdf5_output)
        

        print("\n" + "=" * 60)
        print("Test Loading")
        print("=" * 60)
        embedding_dict = load_embeddings_from_hdf5(hdf5_output)
        print(f"Loaded {len(embedding_dict)} embeddings")
        

        if site_ids:
            first_site_id = site_ids[0]
            if first_site_id in embedding_dict:
                print(f"\nFirst sample:")
                print(f"  site_id: {first_site_id}")
                print(f"  sirna: {embedding_dict[first_site_id]['sirna']}")
                print(f"  embedding shape: {embedding_dict[first_site_id]['embedding'].shape}")
                print(f"  embedding dtype: {embedding_dict[first_site_id]['embedding'].dtype}")
        

        print("\n" + "=" * 60)
        print("Creating Test Dataset")
        print("=" * 60)
        test_dataset = create_simple_dataset_class(csv_file, hdf5_output, sample_size=10)
        
        if len(test_dataset) > 0:
            print(f"\nTest dataset size: {len(test_dataset)}")
            sample = test_dataset[0]
            print(f"Sample data structure:")
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\n✅ All operations completed successfully!")
        print(f"   Output file: {hdf5_output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()