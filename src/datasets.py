"""Dataset related functions and classes"""

import os

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor
from tqdm import tqdm
from collections import defaultdict
from src import constants
from src.clip.clip import tokenize
import random
from src.transformations.cloome import _transform
from functools import partial



class CellPaintingHd5(Dataset):
    """Customized dataset for cell painting images embedding with control embedding generation. for hd5 file"""

    def __init__(
        self,
        sample_index_file: str,
        mole_struc: str = "morgan",
        context_length: int = 77,
        image_directory_path: str = os.path.join(constants.DATASET_DIR, "image_dir"),
        molecule_path: str = os.path.join(
            constants.DATASET_DIR,
            "caption_dir",
        ),
        transforms=None,
        group_views: bool = False,
        subset: float = 1.0,
        num_channels: int = 5,
        unique: bool = False,
        dataset: str = "bray2017",
        cell_type:str="U2SO",
        text_model_path:str='seyonec/ChemBERTa-zinc-base-v1',

        use_true_control:bool=False,
        control_key: str = 'CPD_NAME',
        return_control: bool = True,
        is_test:bool=False,
        use_compound_mean:bool=False,

        use_global_reparam: bool = True,
        global_stats_file: str = None,
        reparam_noise_scale: float = 1.0,
        deterministic_control: bool = False,

        cache_control_embeddings: bool = False,

        control_csv_path: str = None,
    ):
        """Read samples from cellpainting dataset."""

        # Check if the path is a folder or a file (HDF5)
        if os.path.isdir(image_directory_path):
            self.is_hdf5 = False
            assert os.path.exists(
                image_directory_path
            ), f"Image directory {image_directory_path} does not exist."
        elif os.path.isfile(image_directory_path) and image_directory_path.endswith(".h5"):
            self.is_hdf5 = True
            self.h5_path = image_directory_path
            self.img_file = h5py.File(image_directory_path, "r")

            try:
                self.img_ids = [
                    name.decode("utf-8").replace(".npz", "")
                    for name in self.img_file["well_id"][:]
                ]
            except KeyError:

                try:
                    self.img_ids = [
                        name.decode("utf-8").replace(".npz", "")
                        for name in self.img_file["sample_keys"][:]
                    ]
                except KeyError:
                    raise KeyError("HDF5File'well_id''sample_keys'Dataset")
        else:
            raise ValueError(
                "image_directory_path must be either a valid directory or HDF5 file."
            )
        

        self.control_key = "CPD_NAME" if dataset=="bray2017" else "treatment" 
        self.return_control = return_control
        self.use_true_control=use_true_control
        self.is_test=is_test

        self.dataset=dataset
        self.cell_type=cell_type
        self.text_model_path=text_model_path
    
        self.use_compound_mean=use_compound_mean
        

        self.use_global_reparam = use_global_reparam
        self.global_stats_file = global_stats_file
        self.reparam_noise_scale = reparam_noise_scale
        self.deterministic_control = deterministic_control


        self.cache_control_embeddings = cache_control_embeddings
        self.control_embedding_cache = {}
        self.control_csv_path = control_csv_path


        self.compound_to_samples = defaultdict(list)
        self.sample_to_compound = {}
        self.compound_mean_embeddings = {}

        self.control_samples = []


        self.global_mean = None
        self.global_std = None
        self.global_stats_computed = False
        
        # Load sample index
        sample_keys = []
        sample_index = None
        

        if sample_index_file and os.path.exists(sample_index_file):
            print(f"Reading sample index from: {sample_index_file}")
            # Read sample index
            sample_index = pd.read_csv(sample_index_file, sep=",", header=0)
            sample_index.set_index(["SAMPLE_KEY"])
            sample_keys = sample_index['SAMPLE_KEY'].tolist()
            print(f"Found {len(sample_keys)} samples in index file")
            

            for _, row in sample_index.iterrows():
                sample_id = row['SAMPLE_KEY']
                if self.control_key in row and pd.notna(row[self.control_key]):
                    compound = str(row[self.control_key])
                    self.sample_to_compound[sample_id] = compound
                    self.compound_to_samples[compound].append(sample_id)
            
            print(f" {len(self.sample_to_compound)}  samplesCompound")
            print(f" {len(self.compound_to_samples)} Compound")
        

        if image_directory_path is not None:
            self.images = True
            assert os.path.exists(image_directory_path), f"Image directory not found: {image_directory_path}"
            

            image_keys = self._get_image_keys_from_directory(image_directory_path) if not self.is_hdf5 else self.img_ids
            
            print(f"Found {len(image_keys)} images/embeddings in: {image_directory_path}")
            

            if sample_keys:

                image_key_set = set(image_keys)
                sample_key_set = set(sample_keys)
                keys = list(sample_key_set & image_key_set)
                print(f"Intersection with index file: {len(keys)} samples")
            else:

                keys = image_keys
                print(f"Using all {len(keys)} images/embeddings")
            

            sample_keys = keys

        self.image_directory_path = image_directory_path
        self.sample_index = sample_index
        self.group_views = group_views
        self.transforms = transforms
        self.num_channels = num_channels
        self.context_length = context_length
        self.unique = unique

        # Load molecule file
        assert os.path.isfile(
            molecule_path
        ), f"Molecule file {molecule_path} does not exist."

        if mole_struc == "morgan":

            molecule_df = self._load_h5py_molecules(molecule_path)

        elif mole_struc == "text":
            molecule_df = pd.read_csv(molecule_path, index_col=["ID"])
        elif mole_struc == "smiles":

            molecule_df = pd.read_csv(sample_index_file, index_col=["SAMPLE_KEY"])
            # smiles_string = molecule_df.loc[sample_key, "SMILES"]
            # mol=str(smiles_string)



        if unique:  # whether to return only unique treatment.
            if dataset == "rxrx3-core":
                molecule_df["new_index"] = molecule_df.index
            else:
                molecule_df["new_index"] = molecule_df.index.str.rsplit("-", n=1).str[0]

            molecule_df.set_index("new_index", inplace=True)
            molecule_df = molecule_df[~molecule_df.index.duplicated(keep="first")]

        if self.context_length == 256:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
        elif self.context_length == 512:
            self.tokenizer = BertTokenizer.from_pretrained(
                self.text_model_path
            )
        

        self.molecule_df = molecule_df
        self.mole_struc = mole_struc
        # self.unique_molecule = np.unique(molecule_df.values)
        # self.label_map = {
        #     molecule: idx for idx, molecule in enumerate(self.unique_molecule)
        # }
        mol_keys = list(molecule_df.index.values)
        keys = list(set(sample_keys) & set(mol_keys))

        if len(keys) == 0:
            raise Exception("Empty dataset!")

        if subset != 1.0:
            keys = keys[: int(len(keys) * subset)]

        self.keys = keys
        self.n_samples = len(keys)
        self.sample_keys = list(keys)

        # print(f"use control is {self.return_control}")

        if self.return_control:

            if self.use_true_control:
                self._load_control_samples()
                self._load_or_compute_rxrx19a_global_stats()

            elif self.dataset=="rxrx19a":
                if self.use_global_reparam :
                    print(f"StartProcessingInfo...")
                    print(f"use control is {self.return_control} use global is {self.use_global_reparam}")
                    self._load_or_compute_global_stats()

            
        print(f"DatasetSize: {len(self.keys)}  samples")

    def _load_control_samples(self):
        """Load control samples from CSV file (unified for both bray2017 and rxrx19a)."""
        try:
            if self.control_csv_path and os.path.exists(self.control_csv_path):
                control_path = self.control_csv_path
            else:
                control_file = f"{self.cell_type}-control.csv"
                if self.dataset == "rxrx19a":
                    control_path = os.path.join("path/to/your/data/rxrx19a/RxRx19a", control_file)
                elif self.dataset == "bray2017":
                    control_path = os.path.join("path/to/your/data/bray2017", control_file)
                else:
                    control_path = os.path.join("path/to/your/data", control_file)

            if os.path.exists(control_path):
                print(f"LoadcontrolFile: {control_path}")
                control_df = pd.read_csv(control_path)

                if 'SAMPLE_KEY' in control_df.columns:
                    self.control_samples = control_df['SAMPLE_KEY'].tolist()
                    print(f" {len(self.control_samples)} control")

                else:
                    print(f"Warning: controlFileSAMPLE_KEY")
            else:
                print(f"Warning: controlFile: {control_path}")

        except Exception as e:
            print(f"Loadcontrolerror: {e}")

    def _load_or_compute_rxrx19a_global_stats(self):
        """LoadRxRx19aDatasetcontrol"""
        if not self.control_samples:
            print("Warning: control，")
            self.global_stats_computed = False
            return
        
        

        print(f"StartRxRx19a controlInfo...")
        print(f" {len(self.control_samples)} control")
        
        control_embeddings = []
        

        for control_key in tqdm(self.control_samples, desc="controlEmbedding"):
            try:

                if control_key in self.img_ids:
                    index = self.img_ids.index(control_key)
                    embedding = self.img_file["embeddings"][index]
                    control_embeddings.append(embedding)
            except Exception as e:
                print(f"control {control_key} EmbeddingFailed: {e}")
                continue
        
        if len(control_embeddings) > 0:
            control_embeddings_array = np.array(control_embeddings)
            

            self.global_mean = np.mean(control_embeddings_array, axis=0)
            self.global_std = np.std(control_embeddings_array, axis=0)
            

            self.global_std = np.maximum(self.global_std, 1e-8)
            
            self.global_stats_computed = True
            
            print(f" {len(control_embeddings)} controlEmbeddingInfo")
            print(f"MeanShape: {self.global_mean.shape}")
            print(f": min={self.global_std.min():.6f}, max={self.global_std.max():.6f}, mean={self.global_std.mean():.6f}")
            

            # if self.global_stats_file:
            #     try:
            #         np.savez(self.global_stats_file, 
            #                 global_mean=self.global_mean, 
            #                 global_std=self.global_std,
            #                 num_samples=len(control_embeddings),
            #                 cell_line=self.cell_line,
            #                 timestamp=pd.Timestamp.now().isoformat())

            #     except Exception as e:

        else:
            print("Warning：RxRx19aInfo，SuccesscontrolEmbedding")
            self.global_stats_computed = False

    def get_rxrx19a_control_embedding(self):
        """
        RxRx19aDataset：controlParameter
        
        ：z = μ + σ ⊙ ε， ε ∼ N(0, I)
        μσcontrol
        
        Returns:
            control_embedding: controlParameterGenerateEmbedding
        """
        if not self.global_stats_computed or self.global_mean is None or self.global_std is None:
            print("Warning：RxRx19aInfo，GeneratecontrolEmbedding")
            return None
        
        if self.deterministic_control:

            control_embedding = self.global_mean.copy()
        else:

            epsilon = np.random.randn(*self.global_mean.shape).astype(np.float32)
            control_embedding = self.global_mean + self.reparam_noise_scale * self.global_std * epsilon
        
        return control_embedding

    def get_random_control_embedding(self):
        """
        randomcontrolEmbedding
        
        Returns:
            control_embedding: randomcontrolEmbedding
        """
        if not self.control_samples:
            return None
        

        random_control_key = np.random.choice(self.control_samples)
        
        try:
            if random_control_key in self.img_ids:
                index = self.img_ids.index(random_control_key)
                embedding = self.img_file["embeddings"][index].copy()
                return embedding
        except Exception as e:
            print(f"randomcontrol {random_control_key} EmbeddingFailed: {e}")
        
        return None

    def _load_h5py_molecules(self, molecule_file):
        """
        h5pyHDF5FileLoadMolecule
        """
        with h5py.File(molecule_file, 'r') as f:

            if 'fingerprints' not in f or 'sample_keys' not in f:
                raise ValueError(f"Invalid h5py format: missing 'fingerprints' or 'sample_keys' dataset")
            

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
            

            molecule_df = pd.DataFrame.from_dict(fp_dict, orient='index')
            molecule_df.index.name = 'SAMPLE_KEY'
            

            print(f"Loaded {len(molecule_df)} fingerprints from h5py file")
            print(f"Fingerprint shape: {fingerprints.shape}")
            
            return molecule_df


    def _precompute_compound_mean_embeddings(self):
        """CompoundEmbedding"""
        if self.is_hdf5:
            print(f"HDF5File {len(self.compound_to_samples)} CompoundEmbedding...")
            
            for compound, sample_list in tqdm(self.compound_to_samples.items(), desc="Embedding"):

                valid_samples = [s for s in sample_list if s in self.keys]
                if len(valid_samples) >= 2:
                    embeddings = []
                    
                    for sample_key in valid_samples:
                        try:

                            if sample_key in self.img_ids:
                                index = self.img_ids.index(sample_key)
                                embedding = self.img_file["embeddings"][index]
                                embeddings.append(embedding)
                        except Exception:
                            pass
                    

                    if len(embeddings) > 0:

                        mean_embedding = np.mean(embeddings, axis=0)
                        self.compound_mean_embeddings[compound] = mean_embedding
            
            print(f"Success {len(self.compound_mean_embeddings)} CompoundEmbedding")
    

    def _load_or_compute_global_stats(self):
        """LoadEmbeddingMean"""
        if self.global_stats_file and os.path.exists(self.global_stats_file):
            try:
                print(f"FileLoadInfo: {self.global_stats_file}")
                npz_file = np.load(self.global_stats_file, allow_pickle=True)
                self.global_mean = npz_file['global_mean']
                self.global_std = npz_file['global_std']
                self.global_stats_computed = True
                print(f"SuccessLoadInfo")
                print(f"MeanShape: {self.global_mean.shape}")
                print(f"Shape: {self.global_std.shape}")
                return
            except Exception as e:
                print(f"LoadFileFailed: {e}")
                print("Info...")
        

        print(f"StartInfo...")
        
        if self.is_hdf5:

            all_embeddings = []
            sample_count = min(5000, len(self.keys))
            

            if len(self.keys) > sample_count:
                sampled_keys = random.sample(self.keys, sample_count)
            else:
                sampled_keys = self.keys
            
            for key in tqdm(sampled_keys, desc=""):
                try:
                    if key in self.img_ids:
                        index = self.img_ids.index(key)
                        embedding = self.img_file["embeddings"][index]
                        all_embeddings.append(embedding)
                except:
                    continue
            
            if len(all_embeddings) > 0:
                all_embeddings_array = np.array(all_embeddings)
                self.global_mean = np.mean(all_embeddings_array, axis=0)
                self.global_std = np.std(all_embeddings_array, axis=0)
                self.global_stats_computed = True
                
                print(f" {len(all_embeddings)} EmbeddingInfo")
                print(f"MeanShape: {self.global_mean.shape}")
                print(f"Shape: {self.global_std.shape}")
                

                if self.global_stats_file:
                    try:
                        np.savez(self.global_stats_file, 
                                global_mean=self.global_mean, 
                                global_std=self.global_std)
                        print(f"InfoSave: {self.global_stats_file}")
                    except Exception as e:
                        print(f"SaveInfoFailed: {e}")
            else:
                print("Warning：Info，SuccessEmbedding")
                self.global_stats_computed = False
    

    def _get_compound_mean_embedding(self, compound):
        """CompoundEmbedding（）"""

        if compound in self.compound_mean_embeddings:
            return self.compound_mean_embeddings[compound].copy()
        

        elif compound in self.compound_to_samples:
            compound_samples = self.compound_to_samples.get(compound, [])
            embeddings = []
            
            for csample in compound_samples:

                if csample in self.keys and self.is_hdf5:
                    try:
                        if csample in self.img_ids:
                            index = self.img_ids.index(csample)
                            embedding = self.img_file["embeddings"][index]
                            embeddings.append(embedding)
                    except:
                        continue
            
            if len(embeddings) > 0:

                mean_embedding = np.mean(embeddings, axis=0)
                return mean_embedding
        
        return None
    

    def get_global_reparam_control_embedding(self):
        """
        ParameterGenerateEmbedding
        
        ：z = μ + σ ⊙ ε， ε ∼ N(0, I)
        
        Returns:
            control_embedding: ParameterGenerateEmbedding
        """
        if not self.global_stats_computed or self.global_mean is None or self.global_std is None:
            print("Warning：Info，GenerateParameterEmbedding")
            return None
        
        if self.deterministic_control:

            control_embedding = self.global_mean.copy()
        else:

            epsilon = np.random.randn(*self.global_mean.shape).astype(np.float32)
            control_embedding = self.global_mean + self.reparam_noise_scale * self.global_std * epsilon
        
        return control_embedding


    def get_control_embedding(self, idx_or_key=None):
        """
        Embedding（）
        
        Args:
            idx_or_key: 
        
        Returns:
            control_embedding: Embedding
        """

        cache_key = f"control_{idx_or_key}"
        if self.cache_control_embeddings and cache_key in self.control_embedding_cache:
            return self.control_embedding_cache[cache_key].copy()
        
        control_embedding = None
        

        if self.dataset == "rxrx19a":
            if self.use_true_control:
                if self.use_global_reparam and self.global_stats_computed:
                    control_embedding = self.get_rxrx19a_control_embedding()
                else:

                    control_embedding = self.get_random_control_embedding()
            else:
                if self.use_compound_mean:
                    control_embedding = self._get_compound_mean_embedding(self.sample_to_compound(idx_or_key))
                elif self.use_global_reparam:
                    control_embedding = self.get_global_reparam_control_embedding()
                else:

                    if idx_or_key is not None:
                        sample_key = self.keys[idx_or_key] if isinstance(idx_or_key, int) else idx_or_key
                        if sample_key in self.sample_to_compound:
                            compound = self.sample_to_compound[sample_key]
                            control_embedding = self._get_compound_mean_embedding(compound)


        elif self.dataset == "bray2017":
            if self.use_true_control:
                if self.use_global_reparam and self.global_stats_computed:
                    control_embedding = self.get_rxrx19a_control_embedding()
                else:
                    control_embedding = self.get_random_control_embedding()
            elif self.use_compound_mean:
                control_embedding = self._get_compound_mean_embedding(self.sample_to_compound(idx_or_key))
            elif self.use_global_reparam:
                control_embedding = self.get_global_reparam_control_embedding()
            else:

                if idx_or_key is not None:
                    sample_key = self.keys[idx_or_key] if isinstance(idx_or_key, int) else idx_or_key
                    if sample_key in self.sample_to_compound:
                        compound = self.sample_to_compound[sample_key]
                        control_embedding = self._get_compound_mean_embedding(compound)
        

        if control_embedding is not None and self.cache_control_embeddings:
            self.control_embedding_cache[cache_key] = control_embedding.copy()
        
        return control_embedding


    def __len__(self):
        """Return length of sample keys(PLATE-WELL_POSITION_ID)"""
        return len(self.keys)
    
    def _get_image_keys_from_directory(self, image_directory_path):
        """Image directory"""
        image_keys = []
        for filename in os.listdir(image_directory_path):
            if filename.endswith('.npz'):

                sample_key = filename[:-4]
                image_keys.append(sample_key)
        return image_keys

    def __getitem__(self, idx):
        """Return mole and image pairs with optional control embedding"""

        sample_key = self.keys[idx]
        mol_fp = self.molecule_df.loc[sample_key].values


        if self.mole_struc == "text":
            if self.context_length == 256:
                mol = self.tokenizer(mol_fp, context_length=self.context_length)
                mol = torch.squeeze(mol)
            elif self.context_length == 512:
                output = self.tokenizer(
                    mol_fp[0],
                    padding="max_length",
                    max_length=self.context_length,
                    return_tensors="pt",
                )

                # input_ids = output["input_ids"]


                

                # vocab_size = self.tokenizer.vocab_size
                # if input_ids.max() >= vocab_size:


                #     input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                #     output["input_ids"] = input_ids
                mol = {
                    "input_ids": output["input_ids"].squeeze(0),
                    "attention_mask": output["attention_mask"].squeeze(0),
                }
            else:
                mol = tokenize(mol_fp, self.context_length, truncate=True).flatten()
        elif self.mole_struc == "plate":
            # obtain plate id
            match = str(sample_key).split("-")
            mol = match[0] + "-" + match[1]
        elif self.mole_struc == "embedding":
            mol = np.array(eval(self.molecule_df.loc[sample_key].embedding))
            mol = mol.astype(np.float32)
        elif self.mole_struc == "label":
            mol = mol_fp[0]
        elif self.mole_struc ==  "smiles":

            smiles_string = self.molecule_df.loc[sample_key, "SMILES"]
            mol=str(smiles_string)


        elif self.mole_struc == "morgan":
            mol=mol_fp.astype(np.float32)
        else:
            mol = torch.from_numpy(mol_fp)


        img_dict = self.get_image(sample_key)
        img = img_dict["input"]
        

        control_embedding = None
        if self.return_control:
            control_embedding = self.get_control_embedding(idx)
            if control_embedding is None:

                control_embedding = img.copy()


        if self.return_control:
            if self.is_test:
                return (
                    (
                        img,
                        {
                            "channels": np.asarray([c for c in range(img.shape[0])]),
                        },
                    ),
                    mol,
                    control_embedding,
                    sample_key
                )
            else:
                return (
                    (
                        img,
                        {
                            "channels": np.asarray([c for c in range(img.shape[0])]),
                        },
                    ),
                    mol,
                    control_embedding
                )
        else:
            return (
                (
                    img,
                    {
                        "channels": np.asarray([c for c in range(img.shape[0])]),
                    },
                ),
                mol,
            )

    def get_image(self, key):
        """Load cell painting images wrt key"""

        if self.group_views:
            X = self.load_view_group(key)
        else:
            if self.is_hdf5:
                if key in self.img_ids:
                    index = self.img_ids.index(key)
                    X = self.img_file["embeddings"][index]
                else:
                    print(f"ERROR: Missing sample '{key}' in HDF5 file")
                    return dict(input=np.nan, ID=key)
            else:
                filepath = os.path.join(self.image_directory_path, "{}.npz".format(key))
                if os.path.exists(filepath):
                    X = self.load_view(filepath=filepath)
                    if len(X.shape) == 3:
                        X = np.squeeze(X)
                    index = int(np.where(self.sample_index["SAMPLE_KEY"] == key)[0])
                else:
                    print("ERROR: Missing sample '{}'".format(key))
                    return dict(input=np.nan, ID=key)

        if self.transforms:
            X = self.transforms(X)

        return dict(input=X, row_id=index, ID=key)

    def load_view(self, filepath):
        """Load all channels for one sample"""
        npz = np.load(filepath, allow_pickle=True)

        if "sample" in npz:
            image = npz["sample"].astype(np.float32)
            return image

        return None

    def load_view_group(self, groupkey):
        result = np.empty((1040, 2088 - 12, 5), dtype=np.uint8)
        viewgroup = self.sample_index.get_group(groupkey)
        for i, view in enumerate(viewgroup.sort_values("SITE", ascending=True).iterrows()):
            corner = (0 if int(i / 3) == 0 else 520, i % 3 * 692)
            filepath = os.path.join(
                self.image_directory_path, "{}.npz".format(view[1].SAMPLE_KEY)
            )
            v = self.load_view(filepath=filepath)[:, 4:, :]
            result[corner[0] : corner[0] + 520, corner[1] : corner[1] + 692, :] = v
        return result

    def _load_hdf5(self):
        if self.img_file is None:
            self.img_file = h5py.File(self.h5_path, "r", swmr=True)
        return self.img_file

    def get_sample_keys(self):
        return self.sample_keys.copy()
    

    def get_compound_statistics(self):
        """CompoundInfo"""
        compound_counts = {}
        for compound, samples in self.compound_to_samples.items():

            valid_samples = [s for s in samples if s in self.keys]
            if valid_samples:
                compound_counts[compound] = len(valid_samples)
        

        size_distribution = {}
        for count in compound_counts.values():
            if count in size_distribution:
                size_distribution[count] += 1
            else:
                size_distribution[count] = 1
        
        print(f"CompoundCount: {len(compound_counts)}")
        print(f"CompoundCountMinute:")
        for size in sorted(size_distribution.keys()):
            print(f"  Size {size}: {size_distribution[size]} Compound")
        

        compounds_gt_2 = sum(1 for count in compound_counts.values() if count >= 2)
        print(f"≥2CompoundCount: {compounds_gt_2}")
        
        return compound_counts, size_distribution



def get_cellpainting_dataset(args, num_processes, is_train=True,is_test=False,crop_size=336,processor=None, subset=1.0):
    """Helpler function to get cell painting dataloader"""

    # if args.model_type == "classifier":
    #     mole_struc = "plate"
    # elif args.model_type == "pubmed_emb_clip":
    #     mole_struc = "embedding"
    # elif args.model_type in ["vit", "densenet"]:
    #     mole_struc = "label"
    # elif args.model_type in ["cloome", "cloome_phenom1", "molphenix", "cloome_mpnn"]:
    #     mole_struc = "morgan"
    # else:
    #     mole_struc = "text"
    mole_struc=args.mole_struc
    text_model_path=args.text_model_path
    args.is_test = is_test


    if args.dataset == "bray2017":
        if not is_test:
            split_name = (
                f"datasplit{args.split}-{args.dataset_mode}-train" if is_train else f"datasplit{args.split}-{args.dataset_mode}-val"
            )
        else:
            split_name = (
                f"datasplit{args.split}-{args.dataset_mode}-test" if is_test else f"datasplit{args.split}-{args.dataset_mode}-val"
            )

        # sample_index_file = os.path.join(
        #     args.outdir, args.split_label_dir, f"{split_name}.csv"
        # )
        sample_index_file = os.path.join(
            "path/to/your/data/bray2017/metadata", f"{split_name}.csv"
        )
        preprocess_fn = _transform(
            args.image_resolution_train,
            args.image_resolution_val,
            is_train,
            "dataset",
            "crop",
        )
    elif args.dataset=="rxrx19a":
        if not is_test:
            split_name=(f"{args.cell_type}-{args.dataset_mode}-train" if is_train else f"{args.cell_type}-{args.dataset_mode}-val")
        else:
            split_name=(f"{args.cell_type}-{args.dataset_mode}-test")
        sample_index_file=os.path.join(
            "path/to/your/data/rxrx19a/RxRx19a",f"{split_name}.csv"
        )
        preprocess_fn = _transform(
            args.image_resolution_train,
            args.image_resolution_val,
            is_train,
            "dataset",
            "crop",
        )
    elif args.dataset == "jumpcp":
        split_name = "jumpcp_training_label2" if is_train else "jumpcp_testing_label2"
        sample_index_file = os.path.join(f"path_to_jumpcp/{split_name}.csv")

        if args.model_type == "cell_clip":
            preprocess_fn = None
        else:
            preprocess_fn = _transform(
                args.image_resolution_train,
                args.image_resolution_val,
                is_train,
                "dataset",
                "crop",
            )
    elif args.dataset == "rxrx3-core":
        sample_index_file = os.path.join(
            constants.OUT_DIR, "path_to_rxrx3/rxrx3-core_label.csv"
        )
        crop_res = args.image_resolution_train if is_train else args.image_resolution_val
        preprocess_fn = Compose(
            [
                CenterCrop(crop_res),
                ToTensor(),
                Normalize(
                    (47.1314, 40.8138, 53.7692, 46.2656, 28.7243),
                    (24.1384, 23.6309, 28.1681, 23.4018, 28.7255),
                ),
            ]
        )

    # img_dir = os.path.join(constants.DATASET_DIR, args.dataset, args.img_dir)
    img_dir = args.img_dir

    if args.model_type == "long_clip":
        dataset = CellPaintingHd5(
            sample_index_file,
            mole_struc,
            context_length=248,
            transforms=preprocess_fn,
            subset=subset,
            dataset=args.dataset,
            return_control=args.return_control,
            cell_type=args.cell_type,
            use_true_control=args.use_true_control,
            text_model_path=text_model_path,
            is_test=args.is_test,
        )
    elif args.model_type in [
        "bert_clip",
        "pubmed_clip",
        "cloome",
        "clip_channelvit",
        "cell_clip_mae",
        "cloome_mpnn",
    ]:
        if args.model_type in [
            "bert_clip",
            "mil_cell_clip",
            "clip_channelvit",
            "cell_clip_mae",
        ]:
            context_length = 512
        else:
            context_length = 256

        dataset = CellPaintingHd5(
            sample_index_file,
            mole_struc,
            context_length=context_length,
            transforms=preprocess_fn,
            subset=subset,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            cell_type=args.cell_type,
            use_true_control=args.use_true_control,
            text_model_path=text_model_path,
            is_test=args.is_test,
        )
    elif args.model_type in [
        "mil_cell_clip",
        "cell_clip",
        "pheno_cell",
        "cell_sigclip",
        "pubmed_clip_phenom1",
    ]:
        if args.model_type in ["mil_cell_clip", "cell_clip","pheno_cell"]:
            context_length = 512
        else:
            context_length = 256
        dataset = CellPaintingHd5(
            sample_index_file,
            mole_struc,
            context_length=context_length,
            subset=subset,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            cell_type=args.cell_type,
            use_true_control=args.use_true_control,
            text_model_path=text_model_path,
            is_test=args.is_test,
        )
    elif args.model_type in ["cloome_phenom1", "molphenix"]:
        dataset = CellPaintingHd5(
            sample_index_file,
            mole_struc,
            subset=subset,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            cell_type=args.cell_type,
            use_true_control=args.use_true_control,
            text_model_path=text_model_path,
            is_test=args.is_test,
        )
    elif args.model_type in ["mae", "vit"]:
        preprocess_fn = _transform(
            args.image_resolution_train, args.image_resolution_val, is_train, None, "crop"
        )

        dataset = CellPaintingHd5(
            sample_index_file,
            mole_struc,
            transforms=preprocess_fn,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            subset=subset,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            cell_type=args.cell_type,
            use_true_control=args.use_true_control,
            text_model_path=text_model_path,
            is_test=args.is_test,
        )
    else:
        dataset = CellPaintingHd5(
            sample_index_file,
            mole_struc,
            transforms=preprocess_fn,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            subset=subset,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            cell_type=args.cell_type,
            use_true_control=args.use_true_control,
            text_model_path=text_model_path,
            is_test=args.is_test,
        )

    # num_workers = (
    #     4 * torch.cuda.device_count()
    #     if torch.get_num_threads() >= 4
    #     else torch.get_num_threads()
    # )
    num_workers=1
    # Calculate the exact number of full batches per process for distributed training.

    if num_processes > 1:

        adjusted_batch_size = int(args.batch_size / num_processes) * num_processes
        max_length = int(len(dataset) // adjusted_batch_size)
        num_samples = max_length * adjusted_batch_size

        subset_dataset = Subset(dataset, indices=range(num_samples))
    else:
        subset_dataset = dataset
        num_samples = len(dataset)


    current_collate_fn = partial(collate_fn, mole_struc=args.mole_struc)

    if args.model_type == "mil_cell_clip":
        dataloader = DataLoader(
            subset_dataset,
            batch_size=int(args.batch_size / num_processes),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_fn_mil,
        )
    else:
        dataloader = DataLoader(
            subset_dataset,
            batch_size=int(args.batch_size / num_processes),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=is_train,
            # collate_fn=current_collate_fn
        )

    dataloader.num_samples = num_samples
    # dataloader.label_map = dataset.label_map

    return dataloader


def collate_fn_mil(batch):
    """Custom collate function to handle variable-length image sequences."""
    image_tuples, molecules, controls = zip(*batch)  # Unzip batch

    images = [torch.from_numpy(img_tuple[0]) for img_tuple in image_tuples]
    images = torch.stack(images)

    channel_info = [img_tuple[1]["channels"] for img_tuple in image_tuples]


    input_ids = torch.stack([mol["input_ids"] for mol in molecules])
    attention_mask = torch.stack([mol["attention_mask"] for mol in molecules])

    mol_batch = {
        "input_ids": input_ids,  # (B, context_length)
        "attention_mask": attention_mask,  # (B, context_length)
    }



    all_none = all(ctrl is None for ctrl in controls)
    
    if all_none:

        control_batch = None
    else:

        control_list = []
        for ctrl in controls:
            if ctrl is None:

                placeholder = torch.zeros_like(images[0])
                control_list.append(placeholder)
            else:

                if isinstance(ctrl, np.ndarray):
                    control_list.append(torch.from_numpy(ctrl))
                else:

                    control_list.append(ctrl)
        

        try:
            control_batch = torch.stack(control_list)
        except Exception as e:
            print(f"Warning: error: {e}")
            print(f"Shape: {[c.shape if hasattr(c, 'shape') else type(c) for c in control_list]}")

            control_batch = torch.zeros_like(images)

    return (
        (
            images,
            {"channels": channel_info},
        ),
        mol_batch,
        control_batch
    )

def collate_fn(batch, mole_struc):
    """
     collate 。
    ：1. Processing control；2.  smiles 。
    ：image_data  (img_numpy_array, channel_dict) 
    """

    has_control = len(batch[0]) == 3

    if has_control:

        image_tuples, molecules, controls = zip(*batch)

        images, channel_dicts = zip(*image_tuples)
        controls_list = list(controls)
    else:

        image_tuples, molecules = zip(*batch)

        images, channel_dicts = zip(*image_tuples)
        controls_list = None



    images_tensors = []
    for img in images:
        if isinstance(img, np.ndarray):
            images_tensors.append(torch.from_numpy(img))
        elif torch.is_tensor(img):
            images_tensors.append(img)
        else:
            raise TypeError(f"Image has unsupported type: {type(img)}")
    

    images_batch = torch.stack(images_tensors)


    controls_batch = None
    if has_control and controls_list:
        controls_tensors = []
        for ctrl in controls_list:
            if isinstance(ctrl, np.ndarray):
                controls_tensors.append(torch.from_numpy(ctrl))
            elif torch.is_tensor(ctrl):
                controls_tensors.append(ctrl)
            elif ctrl is None:

                if len(images_tensors) > 0:
                    controls_tensors.append(torch.zeros_like(images_tensors[0]))
                else:
                    controls_tensors.append(torch.zeros(5, 3, 224, 224))
            else:
                raise TypeError(f"Control has unsupported type: {type(ctrl)}")
        
        if controls_tensors:
            controls_batch = torch.stack(controls_tensors)


    channel_info = channel_dicts[0] if channel_dicts else None


    if mole_struc == "text":

        input_ids = torch.stack([mol["input_ids"] for mol in molecules])
        attention_mask = torch.stack([mol["attention_mask"] for mol in molecules])
        molecules_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    elif mole_struc == "smiles":
        molecules_batch = list(molecules)



        molecules_batch = list(molecules)
    elif mole_struc == "morgan":

        mol_tensors = []
        for mol in molecules:
            if isinstance(mol, np.ndarray):
                mol_tensors.append(torch.from_numpy(mol))
            elif torch.is_tensor(mol):
                mol_tensors.append(mol)
            else:
                raise TypeError(f"Molecule has unsupported type: {type(mol)}")
        molecules_batch = torch.stack(mol_tensors)
    elif mole_struc in ["plate", "label"]:

        molecules_batch = list(molecules)
    elif mole_struc == "embedding":

        embedding_tensors = []
        for mol in molecules:
            if isinstance(mol, np.ndarray):
                embedding_tensors.append(torch.from_numpy(mol))
            elif torch.is_tensor(mol):
                embedding_tensors.append(mol)
            else:

                try:
                    embedding_array = np.array(eval(mol))
                    embedding_tensors.append(torch.from_numpy(embedding_array))
                except:
                    raise TypeError(f"Embedding has unsupported type: {type(mol)}")
        molecules_batch = torch.stack(embedding_tensors)
    else:

        try:

            mol_tensors = []
            for mol in molecules:
                if isinstance(mol, np.ndarray):
                    mol_tensors.append(torch.from_numpy(mol))
                elif torch.is_tensor(mol):
                    mol_tensors.append(mol)
                else:
                    mol_tensors.append(torch.tensor(mol))
            molecules_batch = torch.stack(mol_tensors)
        except:

            molecules_batch = list(molecules)


    image_data = (images_batch, channel_info)
    
    if has_control:
        return (image_data, molecules_batch, controls_batch)
    else:
        return (image_data, molecules_batch)


def get_data_subset(dataset, n_samples=1000):
    """Returns randomly subsampled dataset. Use for debug purposes."""
    idcs = np.arange(len(dataset))
    n_samples = min(n_samples, len(dataset))
    np.random.shuffle(idcs)  # shuffles inplace
    new_idcs = idcs[:n_samples]

    return Subset(dataset, new_idcs)


def get_mean_std(loader, outfile):
    """Compute mean and standard deviation of images in the dataset."""

    # Initialize tensors to accumulate sum and squared sum
    sum_images = 0.0
    sum_squared_images = 0.0
    num_pixels = 0

    for batch in tqdm(loader):
        (images, extra_tokens), chem = batch

        images = images.to(device="cuda" if torch.cuda.is_available() else "cpu")

        # Compute the number of pixels per batch
        batch_pixels = images.size(0) * images.size(1) * images.size(2)

        # Reshape images to (batch_size * height * width, num_channels)
        images = images.view(-1, images.size(-1))

        # Accumulate the sum and sum of squares
        sum_images += torch.sum(images, dim=0)
        sum_squared_images += torch.sum(images ** 2, dim=0)
        num_pixels += batch_pixels

    # Compute mean and standard deviation
    mean = sum_images / num_pixels
    std = torch.sqrt((sum_squared_images / num_pixels) - (mean ** 2))

    print(mean, std)
    # Save the results to a file
    with open(outfile, "w") as f:
        f.write(f"Mean: {mean.tolist()}\n")
        f.write(f"Std: {std.tolist()}\n")

    return mean, std
