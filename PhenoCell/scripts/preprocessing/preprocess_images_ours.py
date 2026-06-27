"""Preprocessing cell painting images."""
import glob
import io
import itertools
import os
import re
from functools import partial
from multiprocessing import Pool
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from pathlib import Path

# from src.helpler import parallelize

def parallelize(func, iterable, n_workers, **kwargs):
    """Helper function for parallelization"""
    f = partial(func, **kwargs)
    if n_workers > 1:
        with Pool(n_workers) as p:
            results = p.map(f, iterable)
    else:
        results = list(map(f, iterable))
    return results

def illumination_threshold(arr, perc=0.0028):
    """Return threshold value to not display a percentage of highest pixels"""

    perc = perc / 100

    h = arr.shape[0]
    w = arr.shape[1]

    # find n pixels to delete
    total_pixels = h * w
    n_pixels = total_pixels * perc
    n_pixels = int(np.around(n_pixels))

    # find indexes of highest pixels
    flat_inds = np.argpartition(arr, -n_pixels, axis=None)[-n_pixels:]
    inds = np.array(np.unravel_index(flat_inds, arr.shape)).T

    max_values = [arr[i, j] for i, j in inds]

    threshold = min(max_values)

    return threshold


def twentyfour_to_eight_bit(arr, display_max, display_min=0):
    """Convert uint24 to uint8"""

    threshold_image = (arr.astype(float) - display_min) * (arr > display_min)
    

    scaled_image = threshold_image * (256.0 / (display_max - display_min))
    scaled_image[scaled_image > 255] = 255
    

    scaled_image = scaled_image.astype(np.uint8)
    
    return scaled_image


def group_samples(indir):
    """Group images in different site to a single sample"""

    # tiff_files = glob.glob(os.path.join(indir, "*.tif"))
    tiff_files = list(Path(indir).rglob("*.tif"))


    filtered_files = [f for f in tiff_files if "_thumb" not in f.name]


    # pattern = re.compile(r'^[^_]*_[^_]*_[^_]*_w2[^_]*\.tif$', re.IGNORECASE)
    # filtered_files = [f for f in tiff_files if pattern.match(f.name)]


    # filtered_files = [f for f in tiff_files if not re.search(r'_thumb', f.name, re.IGNORECASE)]
    

    sample_groups = {}
    
    for file_path in filtered_files:
        filename = os.path.basename(file_path)
        

        pattern = r'(?P<plate>.+)_(?P<well>[A-Z]\d{2})_s(?P<site>\d+)_(?P<channel>.+)\.tif'
        match = re.match(pattern, filename)
        
        if match:
            
            plate = match.group('plate')
            well = match.group('well')
            site = match.group('site')
            channel = match.group('channel')

            if channel =='Overlay':
                continue
            

            # plate_match = re.search(r'(\d{8}-\d)', plate)
            # if plate_match:
            #     plate_id = plate_match.group(1)
            # else:

            plate_id=plate
            
            sample_id = f"{plate_id}-{well}-{site}"
            
            if sample_id not in sample_groups:
                sample_groups[sample_id] = {}
            

            sample_groups[sample_id][channel] = file_path
            



    sample_list = []
    

    channel_mapping = {
        'Cy5': 'Mito',
        'FITC': 'ER', 
        'YFP': 'RNA',
        'TexasRed': 'AGP',
        'DAPI': 'Hoechst'
    }
    

    ordered_channels = ['Cy5', 'FITC', 'YFP', 'Texas Red', 'DAPI']
    
    for sample_id, channel_files in sample_groups.items():

        # print(f'channel_files:{channel_files}')
	# if all(ch in channel_files for ch in ordered_channels):
        ordered_files = [channel_files[ch] for ch in ordered_channels]
        sample_list.append((sample_id, ordered_files))
    
    return sample_list


def process_sample(sample_data, outdir="."):
    """Aggregate well level sample"""
    sample_id, imglst = sample_data
    

    first_img = Image.open(imglst[0])
    width, height = first_img.size
    first_img.close()
    

    sample = np.zeros((height, width, 5), dtype=np.uint8)
    

    channel_mapping = {
        'Cy5': 'Mito',
        'FITC': 'ER', 
        'YFP': 'RNA',
        'TexasRed': 'AGP',
        'DAPI': 'Hoechst'
    }

    filenames = {}
    channels = {}

    for i, imgfile in enumerate(imglst):

        filename = os.path.basename(imgfile)
        pattern = r'.+_(?P<well>[A-Z]\d{2})_s(?P<site>\d+)_(?P<channel>.+)\.tif'
        matches = re.match(pattern, filename)
        
        if not matches:
            print(f"Cannot parse filename: {filename}")
            continue
            
        channel_short = matches.group("channel")
        channel = channel_mapping.get(channel_short, channel_short)
        
        try:

            arr = np.array(Image.open(imgfile))

            if arr.ndim == 3:
                arr = arr[:, :, 0]
            

            threshold = illumination_threshold(arr)
            scaled_arr = twentyfour_to_eight_bit(arr, threshold)
            

            sample[:, :, i] = scaled_arr
            

            channels[i] = channel
            filenames[channel] = filename
            
        except Exception as e:
            print(f"Cannot process file {imgfile}: {e}")
            return


    os.makedirs(outdir,exist_ok=True)
    outpath = os.path.join(outdir, sample_id)
    np.savez(outpath, sample=sample, channels=channels, filenames=filenames)
    
    print(f"Sample processed: {sample_id}")
    
    return


def get_mean_std(loader, outfile):
    """Compute statistiscs of images."""
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for batch in tqdm(loader):
        images = batch
        images = images["input"]
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    with open(outfile, "w") as f:
        f.write(f"Mean:{mean}\n")
        f.write(f"Std:{std}")

    return mean, std


def get_dataloader(index_file, input_filename_imgs, batch_size):
    """Construct cell painting data loader"""
    assert input_filename_imgs

    dataset = CellPainting(
        index_file,
        input_filename_imgs,
        transforms=ToTensor(),
    )
    num_samples = len(dataset)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_data(args, preprocess_fns):
    """Return data from train and test"""
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data_imgs:
        data["train"] = get_dataloader(args, is_train=True)
    if args.val_data_imgs:
        data["val"] = get_dataloader(args, is_train=False)

    return data


if __name__ == "__main__":

    indir = "path/to/your/data/ours/img/"
    outdir = "path/to/your/data/ours/npz/"
    n_cpus = 60

    index_file = "/<path-to-your-folder>/cellpainting-index.csv"
    input_imgs = "/publicdata/cellpainting/npzs/chembl24"
    input_mols = "/<path-to-your-folder>/morgan_fps_1024.hdf5"
    batchsize = 32


    sample_groups = group_samples(indir)
    print(f" {len(sample_groups)}  samples to process")
    

    result = parallelize(process_sample, sample_groups, n_cpus, outdir=outdir)


    # dataloader = get_dataloader(index_file, input_imgs, batchsize)
    # mean, std = get_mean_std(dataloader, "stats_file.txt")