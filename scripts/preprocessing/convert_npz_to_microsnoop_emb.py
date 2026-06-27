"""Functions to transform cell painting image to clip embeddings"""
import argparse
import os
import random
import re

import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torchvision.transforms import RandomCrop
from torch import nn,optim

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPImageProcessor,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)
import os
import sys
import numpy as np
import cv2 as cv
import pandas as pd
import time
from scellseg.net_utils import downsample as CNN_downsample
from scellseg.net_utils import upsample as CNN_upsample

try:
    import torch
    from torch import optim, nn

    TORCH_ENABLED = True
    torch_GPU = torch.device('cuda')
    torch_CPU = torch.device('cpu')
except:
    TORCH_ENABLED = False
# from src import constants

# from src.helpler import compute_model_stats
# from src.open_phenom.hugginface_mae import MAEModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class makeStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x0):
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** .5
        return style

class CNN_encoder(nn.Module):
    def __init__(self, in_chans=1, depths=[32, 64, 128, 256], sz=3, residual_on=True):
        super().__init__()
        nbase = [in_chans] + depths
        self.downsample = CNN_downsample(nbase, sz=sz, residual_on=residual_on)

    def forward(self, x, mask_ratio=None):  # 'mask_ratio': to harmonise with the vit model
        embeddings = self.downsample(x)
        mask = None  # to harmonise with the vit model
        return embeddings, mask

class CNN_decoder(nn.Module):
    def __init__(self, depths=[32, 64, 128, 256], out_chans=1, sz=3, residual_on=True, concatenation=False):
        super().__init__()
        self.depths = depths
        nbase = depths + [depths[-1]]
        self.upsample = CNN_upsample(nbase, sz=sz, residual_on=residual_on, concatenation=concatenation,
                                     style_channels=[depths[-1], depths[-1], depths[-1], depths[-1]])
        self.make_style = makeStyle()
        self.base_bn = nn.BatchNorm2d(depths[0], eps=1e-5)
        self.base_relu = nn.ReLU(inplace=True)
        self.base_conv = nn.Conv2d(depths[0], out_chans, 1, padding=1 // 2)

    def forward(self, embeddings):
        styles = [self.make_style(embeddings[-1]) for _ in range(len(self.depths))]
        y = self.upsample(styles, embeddings)
        y = self.base_conv(self.base_relu(self.base_bn(y)))
        return y


class CNNNet(nn.Module):
    """
    refer to https://github.com/MouseLand/cellpose and https://github.com/cellimnet/scellseg-publish
    """

    def __init__(self, in_chans=1, out_chans=1, depths=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = CNN_encoder(in_chans=in_chans, depths=depths)
        self.decoder = CNN_decoder(depths=depths, out_chans=out_chans)

    def forward(self, imgs, mask_ratio=None):
        latent, mask = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent)  # [N, L, p*p*3]
        return pred, mask

class CoreModel():
    def __init__(self):
        rank = get_rank()
        if rank==0:
            print(">>>> Model Init Start")

    def load_model(self, checkpoint, resume=False):
        checkpoint_name = None
        if isinstance(checkpoint, str):
            checkpoint_name = checkpoint
            checkpoint = torch.load(checkpoint, map_location='cpu',weights_only=False)
        assert 'model' in checkpoint, 'Please provide correct checkpoint'
        model_dict = checkpoint['model']
        try:
            self.net.load_state_dict(model_dict)
        except:
            model_dict = {k.replace('module.', ''): v for k, v in model_dict.items() if k.startswith('module.')}
            self.net.load_state_dict(model_dict)

        if resume:
            if 'optimizer' in checkpoint and 'checkpoint_epoch' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if checkpoint['checkpoint_epoch'] is not None:
                    self.start_epoch = checkpoint['checkpoint_epoch'] + 1
                    print(">>>> Resume checkpoint %s with optimizer" % str(self.start_epoch))
                if 'scaler' in checkpoint:
                    self.loss_scaler.load_state_dict(checkpoint['scaler'])
                    print(">>>> Resume checkpoint also with loss_scaler!")
        else:
            print(">>>> Successfullly load pre-trained checkpoint", checkpoint_name)

class MicrosnoopCNN(CoreModel):
    def __init__(self, model_init_param, checkpoint=None, input_size=224, patch_size=16):
        super().__init__()
        # self.input_size = model_init_param['input_size']
        # self.patch_size = model_init_param['patch_size']
        self.input_size = input_size
        self.patch_size = patch_size
        self.net = CNNNet(**model_init_param)
        if checkpoint is not None:
            self.load_model(checkpoint=checkpoint)

def load_pretrained_model(checkpoint_path, device='cuda'):

    model_init_param = {
        'in_chans': 1,
        'out_chans': 1,
        'depths': [32, 64, 128, 256]
    }
    

    model = CNNNet(**model_init_param)
    

    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    

    state_dict = checkpoint['model']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"SuccessLoadTraining: {checkpoint_path}")
    return model

class CellPaintingDataset(Dataset):
    """Customized dataset for loading cell painting images"""

    def __init__(
        self, org_files, dataset, file_ids, preprocessor=None, n_crops=5, crop_size=336
    ):
        # self.data_directory = data_directory
        self.org_files=org_files
        self.dataset = dataset

        if dataset == "rxrx3-core":
            file_path_metadata = hf_hub_download(
                "recursionpharma/rxrx3-core",
                filename="metadata_rxrx3_core.csv",
                repo_type="dataset",
            )
            self.metadata = pd.read_csv(file_path_metadata)
            self.file_ids = file_ids
        else:
            self.file_ids = file_ids
        self.n_crops = n_crops
        self.crop_size = crop_size
        self.preprocessor = preprocessor

    def __len__(self):
        """Return length of the dataset"""
        return len(self.file_ids)

    def __getitem__(self, idx):
        """Return item from the dataloader"""
        """idx（name,img） self.org_filesdataloaderHour！！！"""
        name = self.file_ids[idx]
        if self.dataset == "rxrx3-core":
            file_ids = self.metadata[self.metadata.well_id == name].well_id.tolist()
            files = [self.rename_file(f) + ".npz" for f in file_ids]
            if len(files) >= 70:
                files = random.sample(files, 70)
        else:
            files = [f for f in self.org_files if os.path.basename(f).startswith(name)]

        imgs = []

        for filepath in files:
            # filepath = os.path.join(self.data_directory, f)
            image = self.load_view(filepath)
            if image is not None:
                img_crops = self.transform(
                    image, n_crops=self.n_crops, crop_size=self.crop_size
                )
                imgs.extend(img_crops)

        return (name, imgs)

        # for f in files:
        #     filepath = os.path.join(self.data_directory, f)
        #     image = self.load_view(filepath)
        #     if image is not None:
        #         img_crops = self.transform(
        #             image, n_crops=self.n_crops, crop_size=self.crop_size
        #         )
        #         imgs.extend(img_crops)

        # return (name, imgs)

    def load_view(self, filepath):
        """Load cell painting images"""
        npz = np.load(filepath, allow_pickle=True)

        if self.dataset == "rxrx3-core":
            image = npz["images"]
            return image
        else:
            if "sample" in npz:
                image = npz["sample"].astype(np.float32)
                return image

        return None

    def rename_file(self, filename):
        return re.sub(r"_(\d+)_", r"_Plate\1_", filename)

    def transform(self, X, n_crops=5, crop_size=336):
        """Transform cell painting images"""
        transformed_images = []

        if self.dataset == "rxrx3-core":
            channel_order = [1, 3, 4, 0, 2]
            X = X[channel_order, :, :]
            X = np.transpose(X, (1, 2, 0))

        random_crop = RandomCrop(crop_size)

        H, W, num_channels = X.shape
        # For each crop (n_crops times)
        for _ in range(n_crops):
            cropped_img = random_crop(
                torch.tensor(X).permute(2, 0, 1)
            )  # Shape: (num_channels, H, W)

            cropped_img = cropped_img.permute(
                1, 2, 0
            ).numpy()  # Shape: (H, W, num_channels)
            channel_crops = []
            for i in range(num_channels):
                channel_img = cropped_img[:, :, i]

                # Convert grayscale channel to RGB
                channel_rgb = np.repeat(
                    channel_img[:, :, np.newaxis], 3, axis=2
                )  # Shape: (H, W, 3)

                channel_rgb = torch.tensor(channel_rgb, dtype=torch.float32).permute(
                    2, 0, 1
                )  # Shape: (3, H, W)
                if self.preprocessor:
                    channel_crops.append(
                        self.preprocessor(channel_rgb, return_tensors="pt").pixel_values[0]
                    )
                else:
                    channel_crops.append(channel_rgb)

            transformed_images.append(torch.stack(channel_crops))

        transformed_images = torch.stack(transformed_images)

        return transformed_images


def my_collate_fn(batch):
    """Customized collate function, return the list as it is"""
    return batch


def hierarchical_pooling(cell_embeddings, num_clusters=3, weighting="size"):
    """Hierarchical pooling by clustering cells into subpopulations for each channel"""

    _, num_channels, dim = cell_embeddings.shape
    well_embedding = np.zeros((num_channels, dim))

    for c in range(num_channels):
        channel_embeddings = cell_embeddings[:, c, :]

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(channel_embeddings)

        subpop_embeddings = np.zeros((num_clusters, dim))
        subpop_sizes = np.zeros((num_clusters, 1))  # Ensure it has shape (num_clusters, 1)

        np.add.at(subpop_embeddings, cluster_labels, channel_embeddings)
        subpop_sizes[:, 0] = np.bincount(cluster_labels, minlength=num_clusters)

        mask = subpop_sizes[:, 0] > 0  # Extract the 1D boolean mask
        subpop_embeddings[mask] /= subpop_sizes[
            mask
        ]  # Perform element-wise division safely

        if weighting == "size":
            weights = subpop_sizes / np.sum(subpop_sizes)
        else:
            weights = np.ones_like(subpop_sizes) / num_clusters

        well_embedding[c, :] = np.sum(subpop_embeddings * weights, axis=0)

    return well_embedding


def soft_kmeans_pooling(cell_embeddings, num_clusters=3):
    """Perform soft clustering using Gaussian Mixture Model (GMM) and weighted pooling."""
    _, num_channels, dim = cell_embeddings.shape
    well_embedding = np.zeros((num_channels, dim))

    for c in range(num_channels):
        channel_embeddings = cell_embeddings[:, c, :]

        # Soft clustering with GMM (alternative to K-Means)
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        gmm.fit(channel_embeddings)
        soft_assignments = gmm.predict_proba(channel_embeddings)  # Soft cluster assignments

        # Compute cluster means
        subpop_embeddings = np.zeros((num_clusters, dim))
        for k in range(num_clusters):
            subpop_embeddings[k] = np.sum(
                channel_embeddings * soft_assignments[:, k][:, None], axis=0
            )

        subpop_sizes = soft_assignments.sum(axis=0)[:, None]  # Soft counts for each cluster
        mask = subpop_sizes[:, 0] > 0  # Avoid division by zero
        subpop_embeddings[mask] /= subpop_sizes[mask]

        # Weighting based on entropy of assignments
        entropy_weights = -np.sum(
            soft_assignments * np.log(soft_assignments + 1e-8), axis=0
        )[:, None]
        entropy_weights /= np.sum(entropy_weights)

        well_embedding[c, :] = np.sum(subpop_embeddings * entropy_weights, axis=0)

    return well_embedding


def contrastive_pooling(cell_embeddings):
    """Combines max and median pooling for contrastive feature aggregation."""
    _, num_channels, dim = cell_embeddings.shape
    well_embedding = np.zeros((num_channels, dim))

    for c in range(num_channels):
        channel_embeddings = cell_embeddings[:, c, :]

        # Max Pooling (Capture Strongest Features)
        max_pool = np.max(channel_embeddings, axis=0)

        # Median Pooling (Capture Stable Representations)
        median_pool = np.median(channel_embeddings, axis=0)

        # Contrastive Aggregation (Weighted)
        well_embedding[c, :] = 0.5 * max_pool + 0.5 * median_pool  # Equal weighting
        # Alternative: Adaptive weighting based on feature variance

    return well_embedding


# def generate_embeddings(
#     pretrained_clip,
#     batch,
#     output_file,
#     model_card="clip",
#     aggregation=None,
#     max_instances=6,
# ):
#     """Process a batch of images, generate embeddings, and save them"""

#     batch_size = len(batch)

#     for i in range(batch_size):
#         name, image_slides = batch[i][0], batch[i][1]

#         with torch.no_grad():
#             crops_tensor = []

#             for crop_imgs in image_slides:
#                 channel_embeddings = [channel.unsqueeze(0) for channel in crop_imgs]
#                 crops_tensor.append(torch.cat(channel_embeddings, dim=0))

#             crops_tensor = torch.stack(crops_tensor)
#             flattened_tensor = crops_tensor.view(
#                 -1, *crops_tensor.shape[2:]
#             )  # Shape: [num_crops * 5, H, W]

#             if model_card == "openphenom":
#                 embeddings = pretrained_clip.predict(flattened_tensor)
#             else:
#                 embeddings = pretrained_clip(
#                     flattened_tensor
#                 ).pooler_output  # Shape: [num_crops * 5, clip_emb_dim]

#             embeddings = embeddings.view(len(image_slides), 5, -1)

#             if aggregation == "mean":
#                 final_embeddings = torch.mean(embeddings, dim=0).cpu().numpy()
#             elif aggregation == "median":
#                 final_embeddings = torch.median(embeddings, dim=0).values.cpu().numpy()
#             elif aggregation == "hierarchical_pooling":
#                 final_embeddings = hierarchical_pooling(
#                     embeddings.cpu().numpy(), num_clusters=3, weighting="uniform"
#                 )
#             elif aggregation == "gmm":
#                 final_embeddings = soft_kmeans_pooling(
#                     embeddings.cpu().numpy(), num_clusters=3
#                 )
#             else:
#                 instance_num = int(len(image_slides) / 5)
#                 embeddings = embeddings.reshape(
#                     instance_num, 5, 5, -1
#                 )  # (groups, crops, channels, dim)

#                 # Pad with one zero instance if less than max_instance
#                 if embeddings.shape[0] < max_instances:
#                     pad_size = max_instances - instance_num
#                     pad = torch.zeros(
#                         (pad_size, 5, 5, embeddings.shape[-1]),
#                         dtype=embeddings.dtype,
#                         device=embeddings.device,
#                     )
#                     embeddings = torch.cat([embeddings, pad], dim=0)
#                 final_embeddings = torch.mean(embeddings, dim=1).cpu().numpy()

#             save(output_file, final_embeddings, name)
def extract_microsnoop_embeddings(model, batch):
    """
    ImageProcessingEmbedding
    """
    embeddings_list = []
    
    with torch.no_grad():

        for i in range(batch.size(0)):

            single_img = batch[i:i+1]

            # print(f'singel image shape is {single_img.shape}')
            


            gray_img = 0.2989 * single_img[:, 0:1] + 0.5870 * single_img[:, 1:2] + 0.1140 * single_img[:, 2:3]
            

            latent_list, _ = model.encoder(gray_img, mask_ratio=0)
            

            

            embedding = embed_fn(latent_list)
            embeddings_list.append(embedding)
    

    return torch.cat(embeddings_list, dim=0)

def embed_fn(latent):
    embedding = latent[-1]
    embedding = F.avg_pool2d(embedding, kernel_size=(embedding.shape[-2], embedding.shape[-1]))
    flatten = nn.Flatten()
    embedding = flatten(embedding)
    embedding = embedding / torch.sum(embedding ** 2, axis=1, keepdim=True) ** .5
    return embedding

def generate_embeddings(
    pretrained_clip,
    batch,
    output_file,
    model_card="clip",
    aggregation=None,
    max_instances=6,
    # clip_batch_size=512,  # NEW: max crops to process at once
):
    """Memory-efficient batch-processing of images, generate embeddings, and save once."""

    names = []
    all_crops = []
    slide_lengths = []


    # Step 1: Stack crops for all samples
    for name, image_slides in batch:
        names.append(name)
        crops_tensor = torch.stack(image_slides)  # (num_crops, 5, 3, H, W)
        all_crops.append(crops_tensor)
        slide_lengths.append(crops_tensor.shape[0])

    all_crops_tensor = torch.cat(all_crops, dim=0)  # (total_crops, 5, 3, H, W)

    b, c, *img_shape = all_crops_tensor.shape
    flattened_tensor = all_crops_tensor.view(b * c, *img_shape)  # (b * 5, 3, H, W)

    # Step 2: Forward pass in small minibatches
    all_embeddings = []

    with torch.no_grad():
        for batch_start in range(0, flattened_tensor.size(0), 128):
            batch_end = min(batch_start + 128, flattened_tensor.size(0))
            batch = flattened_tensor[batch_start:batch_end]

            if model_card == "openphenom":
                embeddings = pretrained_clip.predict(batch)
            elif model_card =='microsnoop':

                # embeddings = embed_fn(latent)
                embeddings = extract_microsnoop_embeddings(pretrained_clip, batch)
            else:
                embeddings = pretrained_clip(batch).pooler_output

            all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)  # (b * 5, emb_dim)

    # Step 3: Split embeddings back per slide
    final_embeddings_list = []
    start = 0
    for name, num_crops in zip(names, slide_lengths):
        num_crops_total = num_crops * 5
        slide_embeddings = all_embeddings[start : start + num_crops_total]
        start += num_crops_total

        slide_embeddings = slide_embeddings.view(
            num_crops, 5, -1
        )  # (num_crops, 5, emb_dim)

        # Aggregation
        if aggregation == "mean":
            final_embeddings = torch.mean(slide_embeddings, dim=0).cpu().numpy()
        elif aggregation == "median":
            final_embeddings = torch.median(slide_embeddings, dim=0).values.cpu().numpy()
        elif aggregation == "hierarchical_pooling":
            final_embeddings = hierarchical_pooling(
                slide_embeddings.cpu().numpy(), num_clusters=3, weighting="uniform"
            )
        elif aggregation == "gmm":
            final_embeddings = soft_kmeans_pooling(
                slide_embeddings.cpu().numpy(), num_clusters=3
            )
        else:
            instance_num = slide_embeddings.shape[0] // 5
            slide_embeddings = slide_embeddings.view(instance_num, 5, 5, -1)

            if instance_num < max_instances:
                pad_size = max_instances - instance_num
                pad = torch.zeros(
                    (pad_size, 5, 5, slide_embeddings.shape[-1]),
                    dtype=slide_embeddings.dtype,
                    device=slide_embeddings.device,
                )
                slide_embeddings = torch.cat([slide_embeddings, pad], dim=0)

            final_embeddings = torch.mean(slide_embeddings, dim=1).cpu().numpy()

        final_embeddings_list.append(final_embeddings)
    # Step 4: Save everything at once
    save(output_file, final_embeddings_list, names)


def save(output_file, embeddings, well_ids):
    """Save a batch of embeddings with flexible shape to an HDF5 file."""
    embeddings = np.asarray(embeddings)  # (batch_size, ...) could be 3D or 4D
    well_ids = list(well_ids)

    with h5py.File(output_file, "a") as hf:
        batch_shape = embeddings.shape
        batch_size = batch_shape[0]  # first dimension = number of samples
        embedding_shape = batch_shape[1:]  # remaining dimensions (num_channels, emb_dim)
        #  or (num_instances, num_channels, emb_dim)

        if "embeddings" not in hf:
            emb_ds = hf.create_dataset(
                "embeddings",
                data=embeddings,
                maxshape=(None, *embedding_shape),  # Unlimited samples
                chunks=True,
            )
            id_ds = hf.create_dataset(
                "well_id",
                data=np.array(well_ids, dtype=h5py.string_dtype(encoding="utf-8")),
                maxshape=(None,),
                chunks=True,
            )
        else:
            emb_ds = hf["embeddings"]
            id_ds = hf["well_id"]

            old_size = emb_ds.shape[0]

            # Check if new batch shape is consistent with previous embedding shape
            assert (
                emb_ds.shape[1:] == embedding_shape
            ), f"Shape mismatch! Existing embeddings {emb_ds.shape[1:]}"

            # Resize and append new data
            emb_ds.resize((old_size + batch_size, *embedding_shape))
            emb_ds[old_size : old_size + batch_size] = embeddings

            id_ds.resize((old_size + batch_size,))
            id_ds[old_size : old_size + batch_size] = np.array(
                well_ids, dtype=h5py.string_dtype(encoding="utf-8")
            )

def get_all_files(directory, extensions=None):
    """
    Recursively get all file paths under directory (including subdirectories)
    
    Parameter:
    directory: Root directory path
    extensions: File extensions to filter (optional)
    
    Returns:
    FilePATHList
    """
    all_files = []
    

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            

            if extensions:
                if not any(file_path.lower().endswith(ext) for ext in extensions):
                    all_files.append(file_path)
            else:
                all_files.append(file_path)
    
    return all_files

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")
    parser.add_argument("--input_dir", type=str, help="input file directory",default='path/to/your/data/ours/npz/Control-Model-Positive-Yao/Batch2')

    parser.add_argument(
        "--model_card", type=str, help="pretrained model card for embeddings",default='microsnoop'
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="path to directory of datasets.",
        # default='constants.DATASET_DIR',
        default='path/to/your/data/bray2017'
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to be processed",
        default="bray2017",
    )
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        help="dataset to be processed",
    )
    parser.add_argument(
        "--unique",
        help="whether to use unique perturbation.",
        action="store_true",
        default=False,
    )
    parser.add_argument("--n_crop", type=int, help="number of crops", default=5)
    parser.add_argument("--output_file", type=str, help="output file name",default='Batch2_microsnoop_ind.h5')

    return parser.parse_args()


def main(args):
    """Convert images into embeddings"""
    accelerator = Accelerator()

    # Load pre-trained embedding model, e.g. DINOv2 or CLIP.

    if "siglip" in args.model_card:
        pretrained_model = SiglipVisionModel.from_pretrained(args.model_card)
        processor = SiglipImageProcessor.from_pretrained(args.model_card)
        crop_size = processor.size["height"]
    elif "dino" in args.model_card:
        pretrained_model = AutoModel.from_pretrained(args.model_card)
        processor = AutoImageProcessor.from_pretrained(args.model_card)

        if args.model_card == "facebook/dino-vits8":
            crop_size = processor.size["height"]
        else:
            crop_size = processor.crop_size["height"]

    elif "clip" in args.model_card:
        pretrained_model = CLIPVisionModel.from_pretrained(args.model_card)
        processor = CLIPImageProcessor.from_pretrained(args.model_card)
        crop_size = processor.crop_size["height"]
    elif args.model_card == "openphenom":
        pretrained_model = MAEModel.from_pretrained("recursionpharma/OpenPhenom")
        crop_size = 256
        processor = None
    elif args.model_card == 'microsnoop':
        checkpoint_name = \
            r'Microsnoop_cnn_trData-microsnoop_trMode-mae_batchSize-16_inputSize-224_inChans-1_embedDim-256_maskRatio-0.25_epoch-999.pth'
        checkpoint_path = os.path.join('path/to/your/project/microsnoop-publish/output', 'models', checkpoint_name)
        pretrained_model = load_pretrained_model(checkpoint_path)
        processor = None
        crop_size=224

    pretrained_model.eval()
    # print(f'model structure:{pretrained_model.encoder}')
    if processor:
        processor.do_resize = False

    # params, flops = compute_model_stats(
    #     pretrained_model, input_size=(3, crop_size, crop_size)
    # )
    # print(f"Total Parameters: {params:.2f}M")
    # print(f"Total FLOPs: {flops:.2f} GFLOPs")

    output_file = (
        f"path/to/your/data/ours/img_emb/Control-Model-Positive-Yao/{args.output_file}"
    )
    # output_file=args.output_file
    # os.makedirs(output_file)



    # org_files = os.listdir(args.input_dir)
    org_files=get_all_files(args.input_dir,extensions=['.log','.tar'])

    if args.dataset == "rxrx3-core":
        file_path_metadata = hf_hub_download(
            "recursionpharma/rxrx3-core",
            filename="metadata_rxrx3_core.csv",
            repo_type="dataset",
        )
        metadata = pd.read_csv(file_path_metadata)
        file_ids = metadata.well_id.unique().tolist()
    else:
        if args.unique:
            file_ids = list(
                set(
                    [
                        "-".join(file.split("-")[:-1])
                        for file in org_files
                        if file.endswith(".npz")
                    ]
                )
            )
        else:
            file_ids = list(set(os.path.basename(file) for file in org_files if file.endswith(".npz")))

    # print(f'h5 file isfile:{os.path.isfile(output_file)}')
    if os.path.isfile(output_file):
        with h5py.File(output_file, "r") as hf:
            output_ids = [i.decode("utf-8") for i in hf["well_id"][:]]
        file_ids = [id for id in file_ids if id not in output_ids]
        print(f"remaining ids: {len(file_ids)}")

    dataset = CellPaintingDataset(
        # args.input_dir,
        org_files,
        args.dataset,
        file_ids,
        processor,
        n_crops=args.n_crop,
        crop_size=crop_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        collate_fn=my_collate_fn,
        pin_memory=True,
        num_workers=0,
    )

    pretrained_model, dataloader = accelerator.prepare(pretrained_model, dataloader)
    # Iterate over batches of data and generate embeddings
    progress_bar = tqdm(
        range(len(dataloader)),
        initial=0,
        desc="batch",
        disable=not accelerator.is_main_process,
    )

    for batch in dataloader:
        generate_embeddings(
            pretrained_model, batch, output_file, args.model_card, args.aggregation_strategy
        )
        if accelerator.is_main_process:
            progress_bar.update(1)

    print(f"Parallelized {args.model_card} embeddings generation done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
