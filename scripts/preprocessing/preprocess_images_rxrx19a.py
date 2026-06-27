"""Preprocessing cell painting images for HRCE and HUVEC data."""
import glob
import os
import re
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path


def parallelize(func, data, n_cpus, **kwargs):
    """Processing"""
    pool = mp.Pool(n_cpus)
    results = []
    
    for item in data:
        result = pool.apply_async(func, args=(item,), kwds=kwargs)
        results.append(result)
    
    pool.close()
    pool.join()
    
    return [r.get() for r in results]

def group_samples_per_experiment(experiment_path):
    """Group images by site for a specific experiment folder"""
    sample_groups = []
    

    plate_pattern = os.path.join(experiment_path, "Plate*")
    plate_dirs = glob.glob(plate_pattern)
    # print(f"{experiment_path} has samples {len(plate_dirs)}")
    
    for plate_dir in plate_dirs:

        plate_name = os.path.basename(plate_dir)
        plate_num = re.search(r'Plate(\d+)', plate_name)
        if plate_num:
            plate_num = int(plate_num.group(1))
        else:
            plate_num = 1
        

        png_pattern = os.path.join(plate_dir, "*.png")
        png_files = glob.glob(png_pattern)
        # print(f"{plate_dir} has samples {len(png_files)}")
        

        file_dict = {}
        for file_path in png_files:
            filename = os.path.basename(file_path)
            


            pattern = r'(?P<well>[A-Z]{1,2}\d{2})_s(?P<site>\d+)_w(?P<channel>\d+)\.png'
            match = re.match(pattern, filename)
            
            if match:
                well = match.group('well')
                site = int(match.group('site'))
                channel = int(match.group('channel'))
                
                key = (well, site)
                if key not in file_dict:
                    file_dict[key] = {}
                file_dict[key][channel] = file_path
        

        for (well, site), channels_dict in file_dict.items():
            if len(channels_dict) == 5:

                channels = [channels_dict[i] for i in range(1, 6) if i in channels_dict]
                if len(channels) == 5:

                    sample_info = {
                        'paths': channels,
                        'experiment': os.path.basename(experiment_path),
                        'plate_num': plate_num,
                        'well': well,
                        'site': site
                    }
                    sample_groups.append(sample_info)
    
    return sample_groups

def process_sample(sample_info, outdir="."):
    """Process a single sample (5 channels) and save as npz"""
    try:

        cell_type = sample_info['experiment'].split('-')[0]
        cell_output_dir = os.path.join(outdir, cell_type)
        os.makedirs(cell_output_dir, exist_ok=True)
        

        sample_data = []
        for i, img_path in enumerate(sample_info['paths']):

            img = Image.open(img_path)
            

            if img.mode == 'RGB' or img.mode == 'RGBA':
                img = img.convert('L')
            
            arr = np.array(img)
            

            if arr.dtype == np.uint16:

                threshold = illumination_threshold(arr)
                arr = sixteen_to_eight_bit(arr, threshold)
            
            sample_data.append(arr)
        

        sample_stacked = np.stack(sample_data, axis=-1)
        

        output_name = f"{sample_info['experiment']}_{sample_info['plate_num']}_{sample_info['well']}_{sample_info['site']}.npz"
        output_path = os.path.join(cell_output_dir, output_name)
        

        np.savez_compressed(
            output_path,
            sample=sample_stacked,
            experiment=sample_info['experiment'],
            plate_num=sample_info['plate_num'],
            well=sample_info['well'],
            site=sample_info['site']
        )
        
        return True, output_path
    
    except Exception as e:
        print(f"Error processing sample {sample_info}: {e}")
        return False, str(e)

def illumination_threshold(arr, perc=0.0028):
    """Return threshold value to not display a percentage of highest pixels"""
    perc = perc / 100
    h, w = arr.shape
    total_pixels = h * w
    n_pixels = int(np.around(total_pixels * perc))
    
    # find indexes of highest pixels
    flat_inds = np.argpartition(arr, -n_pixels, axis=None)[-n_pixels:]
    inds = np.array(np.unravel_index(flat_inds, arr.shape)).T
    max_values = [arr[i, j] for i, j in inds]
    
    return min(max_values) if max_values else arr.max()

def sixteen_to_eight_bit(arr, display_max, display_min=0):
    """Convert unit16 to unit8"""
    threshold_image = (arr.astype(float) - display_min) * (arr > display_min)
    scaled_image = threshold_image * (256.0 / (display_max - display_min))
    scaled_image[scaled_image > 255] = 255
    return scaled_image.astype(np.uint8)

def process_all_experiments(input_dir, output_dir, n_cpus=None):
    """Process all experiment folders in the input directory"""

    os.makedirs(output_dir, exist_ok=True)
    

    experiment_pattern = os.path.join(input_dir, "*")
    experiment_dirs = glob.glob(experiment_pattern)
    experiment_dirs = [d for d in experiment_dirs if os.path.isdir(d)]
    
    print(f"Found {len(experiment_dirs)} experiment folders:")
    for exp in experiment_dirs:
        print(f"  - {os.path.basename(exp)}")
    
    all_sample_groups = []
    

    for experiment_dir in experiment_dirs:
        print(f"\nProcessing experiment: {os.path.basename(experiment_dir)}")
        sample_groups = group_samples_per_experiment(experiment_dir)
        print(f"  Found {len(sample_groups)} complete samples (5 channels each)")
        all_sample_groups.extend(sample_groups)
    
    print(f"\nTotal samples to process: {len(all_sample_groups)}")
    

    if n_cpus is None:
        n_cpus = max(1, mp.cpu_count() - 2)
    

    results = parallelize(process_sample, all_sample_groups, n_cpus, outdir=output_dir)
    

    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    
    print(f"\nProcessing completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    

    if failed > 0:
        print("\nFailed files:")
        for success, result in results:
            if not success:
                print(f"  - {result}")

if __name__ == "__main__":

    input_dir = "path/to/your/data/RXRX19a/image/RxRx19a/images"
    output_dir = "path/to/your/data/RXRX19a/processed_npzs"
    

    process_all_experiments(input_dir, output_dir, n_cpus=60)
    
    print(f"\nProcessing complete! NPZ files saved in:")
    print(f"  {output_dir}/HRCE/")
    print(f"  {output_dir}/VERO/")