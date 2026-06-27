#!/bin/bash

################### Slurm Job Configurations ###################

#SBATCH --job-name=dinov2-HRCE
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

#SBATCH --output=./scripts/preprocessing/%x_%j.out
#SBATCH --error=./scripts/preprocessing/%x_%j.err



################### Your Job Logic Below ###################
input_dir='path/to/your/data/RXRX19a/processed_npzs/HRCE'
output_file='dinov2-giant_ind_HRCE_1536.h5'
csv_file='path/to/your/data/RXRX19a/RxRx19a'
hdf5_output='path/to/your/data/RXRX19a/embeddings/HRCE_morgan_chiral_fps_1024.hdf5'
cell_type='HRCE'
caption_type='cell_captions'
dataset='bray2017'

python scripts/preprocessing/convert_npz_to_avg_emb.py --input_dir $input_dir --output_file $output_file