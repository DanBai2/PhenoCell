"""
Main script for training treatment-images pair for cell painting

Mainly adopted from open_clip[1]

[1] https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/main.py
[2] https://amsword.medium.com/gradient-backpropagation-with-torch-distributed
"""

import argparse
import glob
import os
import time
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from networkx import predecessor

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from vendor.hopfieldlayers.hflayers import Hopfield
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
# import wandb
from src import constants
from src.clip.clip import load_model
from src.clip.methods import (
    bi_cwcl_loss,
    clip,
    cloob,
    cwcl_loss,
    cwcl_ma_loss,
    s2l_loss,
    sigmoid_loss,
)
from src.datasets import CellPaintingImg
from src.helpler import (
    all_gather,
    compute_grad_norm,
    compute_param_norm,
    get_max_steps,
    get_metrics,
    print_args,
)
from src.scheduler import (
    const_lr,
    const_lr_cooldown,
    cosine_lr,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor
from tqdm import tqdm
from transformers import BertTokenizer
from collections import OrderedDict, defaultdict
# from configs.data_config import DataAugmentationConfig
from src import constants
from src.clip.clip import tokenize
import random
# from src.transformations.cell import CellAugmentation
from src.transformations.cloome import _transform
from torchvision.transforms import RandomCrop
torch.backends.cuda.matmul.allow_tf32 = True
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPImageProcessor,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)

def setup_logging(output_dir, name):
    """Setup logging system"""

    os.makedirs(output_dir, exist_ok=True)
    

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    

    logger.handlers.clear()
    

    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")

    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUT_DIR
    )
    parser.add_argument(
        "--split_label_dir",
        type=str,
        help="output parent directory",
        default=constants.SPLIT_LABEL_DIR,
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="index of dataset split file",
    )
    parser.add_argument(
        "--is_train",
        help="whether to use training index",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--return_control",
        help="whether to use control emb",
        action="store_true",
        default=False,
    )   
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training",
        default=42,
    )
    parser.add_argument(
        "--wandb",
        help="whether to monitor model training with wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--wandb_id",
        help="id for monitor training if laod from checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pheno_cell",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default="facebook/dinov2-giant",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--embedding_name",
        type=str,
        default=None,
        help=("image embeddings, e.g. siglip@224, clip@224."),
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="path/to/your/data/bray2017/all_imgages",
        help=("Path to training input directory."),
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        help="Dimension of input emebddings.",
        default=1536,
    )
    parser.add_argument(
        "--molecule_path",
        type=str,
        default="path/to/your/data/cell_long_captions_all.csv",
        help=("Path to molecule (text) data."),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bray2017",
        help=("dataset name, e.g. bray2017 or jumpcp"),
    )
    parser.add_argument(
        "--unique",
        help="whether to use unique perturbation.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="clip",
        help=("Loss types, e.g. cloob, clip."),
    )
    parser.add_argument(
        "--pretrained",
        help="whether to use pretrained text encoder from CLIP.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resume",
        help="whether to use resume from previous training.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fine_tune_ckpt",
        help="path to ckpt for fine tuning.",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="training epochs",
        default=500,
    )
    parser.add_argument(
        "--image_resolution_train",
        default=224,
        nargs="+",
        type=int,
        help="resolution for training set ",
    )
    parser.add_argument(
        "--image_resolution_val",
        default=224,
        nargs="+",
        type=int,
        help="resolution for validation set ",
    )
    parser.add_argument(
        "--val_subset_ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help=(
            "Training batch size. When training with Accelerate, "
            "the batch size passed to the dataloader is the batch size per GPU"
        ),
        default=32,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5.0e-4,
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            "LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown'"
            " (constant w/ cooldown). Default: cosine"
        ),
    )
    parser.add_argument("--eps", type=float, default=1.0e-8, help="Adam epsilon.")
    parser.add_argument(
        "--epochs-cooldown",
        type=int,
        default=None,
        help=(
            "When scheduler w/ cooldown used, "
            "perform cooldown from total_epochs - cooldown_epochs onwards."
        ),
    )
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=1000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--num_cycles", type=int, default=5, help="Number of cosine cycle during training."
    )

    # CLIP temperature
    parser.add_argument(
        "--init-inv-tau", type=float, default=14.3, help="Initial inverse tau."
    )
    parser.add_argument(
        "--learnable-inv-tau",
        default=False,
        action="store_true",
        help="Use a trainable logit scale for the nce loss.",
    )
    # Cloome hopfield params
    parser.add_argument(
        "--scale_hopfield", type=float, default=14.3, help="Scale for Hopfield retrieval."
    )
    parser.add_argument(
        "--learnable-scale-hopfield",
        default=False,
        action="store_true",
        help="Use a trainable logit scale for the Hopfield retrieval.",
    )
    parser.add_argument(
        "--ckpt_freq", type=int, default=10, help="How often to save checkpoints (in epochs)."
    )
    parser.add_argument(
        "--log_freq", type=int, default=100, help="How often to log training info (in batches)."
    )
    parser.add_argument(
        "--eval_freq", type=int, default=1, help="How often to evaluate model training (in epochs)."
    )
    parser.add_argument(
        "--keep_all_ckpts",
        help="whether to keep all the checkpoints",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main(args):
    """Training scripts for contrastive learning."""
    set_seed(args.opt_seed)  # Seed for model optimization.

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_outdir = os.path.join(
        constants.OUT_DIR,
        "results",
        args.dataset,
        "models",
        args.model_type,
        f"{args.model_type}_split{args.split}_train{args.is_train}_epochs{args.epochs}_{timestamp}"
    )
    
    if accelerator.is_main_process:

        os.makedirs(model_outdir, exist_ok=True)
        

        logger = setup_logging(model_outdir, 'training')
        
        logger.info("=" * 80)
        logger.info(f"Training Configuration")
        logger.info("=" * 80)
        for arg in vars(args):
            logger.info(f"{arg}: {getattr(args, arg)}")
        logger.info("=" * 80)
        logger.info(f"Number of GPU available: {torch.cuda.device_count()}")
    else:

        logger = None
    

    print(f"Process {accelerator.process_index} started")
    
    if "siglip" in args.model_card:
        pretrained_model = SiglipVisionModel.from_pretrained(args.model_card)
        processor = SiglipImageProcessor.from_pretrained(args.model_card)
        crop_size = processor.size["height"]
    elif "dino" in args.model_card:
        pretrained_model = AutoModel.from_pretrained(args.model_card)
        processor = AutoImageProcessor.from_pretrained(args.model_card)

        if args.model_card == "facebook/dino-vitb8":
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
    
    # Obtain training & evaluation data
    train_dataloader = get_cellpainting_dataset(
        args,
        accelerator.num_processes,
        is_train=args.is_train,
        crop_size=crop_size,
        processor=processor
    )
    eval_dataloader = get_cellpainting_dataset(
        args,
        accelerator.num_processes,
        is_train=False,
        crop_size=crop_size,
        processor=processor,
        subset=args.val_subset_ratio,
    )
    
    if accelerator.is_main_process:
        logger.info(
            f"Initialize training and eval loader. Number of samples: "
            f"train:{train_dataloader.num_samples}, eval:{eval_dataloader.num_samples}"
        )
        
    # Initialize model
    model = load_model(
        args.model_type,
        args.pretrained,
        args.image_resolution_train,
        vision_width=args.input_dim,
        loss_type=args.loss_type,
        model_card=args.model_card
    )

    exclude = (
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )

    def include(n, p):
        return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )


    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epochs

    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const-cooldown":
        assert (
            args.epochs_cooldown is not None
        ), "Please specify the number of cooldown epochs for this lr schedule."
        cooldown_steps = steps_per_epoch * args.epochs_cooldown
        scheduler = const_lr_cooldown(
            optimizer,
            args.lr,
            args.warmup,
            total_steps,
            cooldown_steps,
            args.lr_cooldown_power,
            args.lr_cooldown_end,
        )
    elif args.lr_scheduler == "cosine-restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            warmup=args.warmup,
            num_cycles=args.num_cycles,
            num_training_steps=total_steps,
        )
    elif args.lr_scheduler == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.1,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    else:
        raise ValueError(
            f"Unknown scheduler, {args.lr_scheduler}. "
            f"Available options are: cosine, const, const-cooldown."
        )

    epoch = 0
    total_steps_done = 0
    total_training_time = 0
    

    history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': [],
        'lr': [],
        'epochs': []
    }
    

    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    best_r1_score = -1.0
    best_r1_epoch = 0
    best_r1_metrics = None

    if args.fine_tune_ckpt:
        try:
            ckpt = torch.load(args.fine_tune_ckpt, map_location="cpu")
            if accelerator.is_main_process:
                logger.info(f"Loading pretrained checkpoint at {args.fine_tune_ckpt}")
        except RuntimeError:
            if accelerator.is_main_process:
                logger.warning(f"Pretrained check point at {args.fine_tune_ckpt} does not exist.")

    elif args.resume:
        existing_steps = get_max_steps(model_outdir)
        if existing_steps is not None:
            ckpt_path = os.path.join(model_outdir, f"ckpt_epoch_{existing_steps:0>4}.pt")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                total_steps_done = ckpt.get("steps", 0)
                epoch = ckpt.get("epoch", 0)
                total_training_time = ckpt.get("total_training_time", 0)
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                if accelerator.is_main_process:
                    logger.info(f"Resuming from epoch {epoch}, steps {total_steps_done}")
            except RuntimeError:
                if accelerator.is_main_process:
                    logger.warning(f"Check point at {ckpt_path} does not exist.")

    if args.loss_type == "cloob":
        hopfield_layer = Hopfield(
            input_size=512,
            scaling=args.scale_hopfield,
            normalize_hopfield_space=False,
            normalize_hopfield_space_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            state_pattern_as_static=True,
            pattern_projection_as_static=True,
            stored_pattern_as_static=True,
            disable_out_projection=True,
            num_heads=1,
            dropout=False,
        )
        model, hopfield_layer = accelerator.prepare(model, hopfield_layer)
    else:
        model = accelerator.prepare(model)

    loss_fct_img = nn.CrossEntropyLoss()
    loss_fct_tx = nn.CrossEntropyLoss()

    optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        optimizer, scheduler, train_dataloader, eval_dataloader
    )


    for current_epoch in range(epoch, args.epochs):
        epoch_start_time = time.time()
        
        if accelerator.is_main_process:
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {current_epoch + 1}/{args.epochs}")
            logger.info(f"{'='*80}")
        

        model.train()
        train_losses = []
        train_batch_times = []
        

        epoch_progress = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {current_epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process,
        )
        
        for batch_idx, batch in epoch_progress:
            batch_start_time = time.time()
            optimizer.zero_grad()

            if args.return_control:
                (images, extra_tokens), treatments, controls = batch
            else:
                (images, extra_tokens), treatments = batch 
                controls = None
            
            # images = images.to(device)
            # print(f"batch size is {len(batch)}")
            # print(f"image length is {len(images)}")
            
            if controls is not None:
                if isinstance(controls, tuple):
                    controls = tuple(
                        item.to(device) if hasattr(item, 'to') else item 
                        for item in controls
                    )
                elif hasattr(controls, 'to'):
                    controls = controls.to(device)
                elif isinstance(controls, np.ndarray):
                    controls = torch.from_numpy(controls).to(device)
            
            m = model.module if accelerator.use_distributed else model

            if args.model_type in ["molphenix", "cloome_mpnn"]:
                text_features = model.encode_mols(treatments.to(device))
            elif args.model_type in [
                "bert_clip",
                "clip_channelvit",
                "cell_clip_mae",
                "cell_clip",
                "pheno_cell",
                "mil_cell_clip",
            ]:
                treatments = {k: v.to(device) for k, v in treatments.items()}
            else:
                treatments = treatments.to(device)

            if args.model_type == "cell_clip" or args.model_type=="pheno_cell":
                images = m.encode_mil(images)
                if controls is not None:
                    controls = m.encode_mil(controls)

            if args.model_type == "clip_channelvit":
                img_features, text_features, logit_scale = model(
                    images, extra_tokens, treatments
                )
            else:
                if args.loss_type in ["s2l", "sigclip"]:
                    img_features, text_features, logit_scale, bias = model(
                        images, treatments, controls
                    )
                else:
                    img_features, text_features, logit_scale = model(images, treatments, controls)

            if accelerator.use_distributed:
                all_image_features = all_gather(img_features)
                all_text_features = all_gather(text_features)
                all_images = all_gather(images)
            else:
                all_image_features = img_features
                all_text_features = text_features
                all_images = images


            if args.loss_type == "clip":
                loss = clip(
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    loss_fct_img,
                    loss_fct_tx,
                )
            elif args.loss_type == "cwcl":
                loss = cwcl_loss(
                    all_images,
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    loss_fct_tx,
                )
            elif args.loss_type == "bi_cwcl":
                loss = bi_cwcl_loss(
                    all_images,
                    all_image_features,
                    all_text_features,
                    logit_scale,
                )
            elif args.loss_type == "cwcl_ma":
                loss = cwcl_ma_loss(
                    all_images,
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    loss_fct_tx,
                )
            elif args.loss_type == "cloob":
                loss = cloob(
                    all_image_features,
                    all_text_features,
                    logit_scale.exp(),
                    hopfield_layer,
                )
            elif args.loss_type == "sigclip":
                loss = sigmoid_loss(
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    bias,
                )
            elif args.loss_type == "s2l":
                loss = s2l_loss(
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    bias,
                )
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 20.0)
            
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)
            

            train_losses.append(loss.detach().cpu().item())
            batch_time = time.time() - batch_start_time
            train_batch_times.append(batch_time)
            total_steps_done += 1
            

            if (batch_idx + 1) % args.log_freq == 0 or batch_idx == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = np.mean(train_losses[-args.log_freq:]) if len(train_losses) >= args.log_freq else train_losses[-1]
                
                if accelerator.is_main_process:
                    logger.info(
                        f"Epoch {current_epoch + 1}/{args.epochs} | "
                        f"Batch {batch_idx + 1}/{len(train_dataloader)} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Temp: {m.logit_scale.data.exp():.4f}"
                    )
                

                epoch_progress.set_postfix({
                    'loss': avg_loss,
                    'lr': f'{current_lr:.2e}',
                    'temp': f'{m.logit_scale.data.exp():.2f}'
                })
        

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = np.mean(train_losses)
        avg_batch_time = np.mean(train_batch_times) if train_batch_times else 0
        total_training_time += epoch_time
        
        if accelerator.is_main_process:
            logger.info(f"\nEpoch {current_epoch + 1} Summary:")
            logger.info(f"  Training Loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            logger.info(f"  Avg Batch Time: {avg_batch_time:.3f}s")
            logger.info(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")
            logger.info(f"  Temperature: {m.logit_scale.data.exp():.4f}")
        

        history['train_loss'].append(avg_epoch_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['epochs'].append(current_epoch + 1)
        

        if (current_epoch + 1) % args.ckpt_freq == 0 and accelerator.is_main_process:
            if not args.keep_all_ckpts:
                pattern = os.path.join(model_outdir, "ckpt_epoch_*.pt")
                for filename in glob.glob(pattern):
                    os.remove(filename)
            
            checkpoint_path = os.path.join(model_outdir, f"ckpt_epoch_{current_epoch + 1:04d}.pt")
            torch.save(
                {
                    "model": accelerator.get_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "steps": total_steps_done,
                    "epoch": current_epoch + 1,
                    "total_training_time": total_training_time,
                    "train_loss": avg_epoch_loss,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        

        if (current_epoch + 1) % args.eval_freq == 0 or current_epoch == 0 or current_epoch == args.epochs - 1:
            model.eval()
            
            if accelerator.is_main_process:
                logger.info(f"\nEvaluating at Epoch {current_epoch + 1}...")
            
            with torch.no_grad():
                all_eval_image_features = []
                all_eval_text_features = []
                all_images = []
                val_losses = []
                
                eval_progress = tqdm(
                    enumerate(eval_dataloader),
                    total=len(eval_dataloader),
                    desc=f"Validation Epoch {current_epoch + 1}",
                    disable=not accelerator.is_main_process,
                )
                
                for eval_batch_idx, batch in eval_progress:
                    if args.return_control:
                        (images, extra_tokens), treatments, controls = batch
                    else:
                        (images, extra_tokens), treatments = batch 
                        controls = None
                    
                    images = images.to(device)
                    
                    if controls is not None:
                        if isinstance(controls, tuple):
                            controls = tuple(
                                item.to(device) if hasattr(item, 'to') else item 
                                for item in controls
                            )
                        elif hasattr(controls, 'to'):
                            controls = controls.to(device)
                        elif isinstance(controls, np.ndarray):
                            controls = torch.from_numpy(controls).to(device)
                    
                    m = model.module if accelerator.use_distributed else model
                    
                    if args.model_type in ["molphenix", "cloome_mpnn"]:
                        text_features = model.encode_mols(treatments.to(device))
                    elif args.model_type in [
                        "bert_clip",
                        "clip_channelvit",
                        "cell_clip_mae",
                        "cell_clip",
                        "mil_cell_clip",
                    ]:
                        treatments = {k: v.to(device) for k, v in treatments.items()}
                    else:
                        treatments = treatments.to(device)
                    
                    if args.model_type == "cell_clip" or args.model_type=="pheno_cell":
                        images = m.encode_mil(images)
                        if controls is not None:
                            controls = m.encode_mil(controls)
                    
                    if args.model_type == "clip_channelvit":
                        img_features, text_features, logit_scale = model(
                            images, extra_tokens, treatments
                        )
                    else:
                        if args.loss_type in ["s2l", "sigclip"]:
                            img_features, text_features, logit_scale, bias = model(
                                images, treatments, controls
                            )
                        else:
                            img_features, text_features, logit_scale = model(
                                images, treatments, controls
                            )
                    
                    all_eval_image_features.append(img_features)
                    all_eval_text_features.append(text_features)
                    all_images.append(images)
                    

                    if args.loss_type == "clip":
                        val_loss = clip(
                            img_features,
                            text_features,
                            logit_scale,
                            loss_fct_img,
                            loss_fct_tx,
                        )
                    elif args.loss_type == "cwcl":
                        val_loss = cwcl_loss(
                            images,
                            img_features,
                            text_features,
                            logit_scale,
                            loss_fct_tx,
                        )
                    elif args.loss_type == "bi_cwcl":
                        val_loss = bi_cwcl_loss(
                            images,
                            img_features,
                            text_features,
                            logit_scale,
                        )
                    elif args.loss_type == "cwcl_ma":
                        val_loss = cwcl_ma_loss(
                            images,
                            img_features,
                            text_features,
                            logit_scale,
                            loss_fct_tx,
                        )
                    elif args.loss_type == "sigclip":
                        val_loss = sigmoid_loss(
                            img_features,
                            text_features,
                            logit_scale,
                            bias,
                        )
                    elif args.loss_type == "s2l":
                        val_loss = s2l_loss(
                            img_features,
                            text_features,
                            logit_scale,
                            bias,
                        )
                    elif args.loss_type == "cloob":
                        val_loss = cloob(
                            img_features,
                            text_features,
                            logit_scale.exp(),
                            hopfield_layer,
                        )
                    
                    val_losses.append(val_loss.detach().cpu().item())
                

                all_eval_image_features = torch.cat(all_eval_image_features)
                all_eval_text_features = torch.cat(all_eval_text_features)
                all_images = torch.cat(all_images)
                
                if accelerator.use_distributed:
                    all_eval_image_features = accelerator.gather_for_metrics(
                        all_eval_image_features
                    )
                    all_eval_text_features = accelerator.gather_for_metrics(
                        all_eval_text_features
                    )
                    all_images = accelerator.gather_for_metrics(all_images)
                

                avg_val_loss = np.mean(val_losses)
                

                if accelerator.is_main_process:
                    metrics = get_metrics(
                        all_eval_image_features, all_eval_text_features
                    )
                    

                    history['val_loss'].append(avg_val_loss)
                    history['metrics'].append(metrics)
                    

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_val_loss_epoch = current_epoch + 1
                        

                        best_loss_path = os.path.join(model_outdir, "best_model_loss.pt")
                        torch.save(
                            {
                                "model": accelerator.get_state_dict(model),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "epoch": current_epoch + 1,
                                "val_loss": avg_val_loss,
                                "metrics": metrics,
                            },
                            best_loss_path,
                        )
                        logger.info(f"✓ New best validation loss: {avg_val_loss:.4f} (epoch {current_epoch + 1})")
                    

                    current_r1 = metrics.get('image_to_text_R@1', 0.0)
                    if current_r1 > best_r1_score:
                        best_r1_score = current_r1
                        best_r1_epoch = current_epoch + 1
                        best_r1_metrics = metrics.copy()
                        

                        best_r1_path = os.path.join(model_outdir, "best_model_r1.pt")
                        torch.save(
                            {
                                "model": accelerator.get_state_dict(model),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "epoch": current_epoch + 1,
                                "val_loss": avg_val_loss,
                                "metrics": metrics,
                            },
                            best_r1_path,
                        )
                        logger.info(f"✓ New best R@1: {current_r1:.4f} (epoch {current_epoch + 1})")
                    

                    logger.info(f"\nEvaluation Results - Epoch {current_epoch + 1}:")
                    logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
                    logger.info(f"  Image→Text R@1: {current_r1:.4f}")
                    logger.info(f"  Image→Text R@5: {metrics.get('image_to_text_R@5', 0.0):.4f}")
                    logger.info(f"  Image→Text R@10: {metrics.get('image_to_text_R@10', 0.0):.4f}")
                    logger.info(f"  Text→Image R@1: {metrics.get('text_to_image_R@1', 0.0):.4f}")
                    logger.info(f"  Text→Image R@5: {metrics.get('text_to_image_R@5', 0.0):.4f}")
                    logger.info(f"  Text→Image R@10: {metrics.get('text_to_image_R@10', 0.0):.4f}")
    

    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        

        history_path = os.path.join(model_outdir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        

        final_model_path = os.path.join(model_outdir, "final_model.pt")
        torch.save(
            {
                "model": accelerator.get_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": args.epochs,
                "total_training_time": total_training_time,
                "history": history,
            },
            final_model_path,
        )
        logger.info(f"Final model saved to {final_model_path}")
        

        logger.info("\n" + "="*80)
        logger.info("BEST MODEL PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Best Validation Loss: {best_val_loss:.4f} (epoch {best_val_loss_epoch})")
        logger.info(f"Best Image→Text R@1: {best_r1_score:.4f} (epoch {best_r1_epoch})")
        
        if best_r1_metrics:
            logger.info("\nBest R@1 Model Full Metrics:")
            for key, value in best_r1_metrics.items():
                logger.info(f"  {key}: {value:.6f}")
        
        logger.info("\n" + "="*80)
        logger.info("Training Statistics:")
        logger.info(f"Total Training Time: {total_training_time:.2f}s")
        logger.info(f"Total Steps: {total_steps_done}")
        logger.info(f"Total Epochs: {args.epochs}")
        logger.info(f"Final Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        logger.info("="*80)
    
def get_cellpainting_dataset(args, num_processes, is_train=True,crop_size=336,processor=None, subset=1.0):
    """Helpler function to get cell painting dataloader"""

    if args.model_type == "classifier":
        mole_struc = "plate"
    elif args.model_type == "pubmed_emb_clip":
        mole_struc = "embedding"
    elif args.model_type in ["vit", "densenet"]:
        mole_struc = "label"
    elif args.model_type in ["cloome", "cloome_phenom1", "molphenix", "cloome_mpnn"]:
        mole_struc = "morgan"
    else:
        mole_struc = "text"

    if args.dataset == "bray2017":
        split_name = (
            f"datasplit{args.split}-train" if is_train else f"datasplit{args.split}-val"
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
        dataset = CellPaintingImg(
            sample_index_file,
            mole_struc,
            context_length=248,
            transforms=preprocess_fn,
            subset=subset,
            dataset=args.dataset,
            return_control=args.return_control,
            crop_size=crop_size,
            processor=processor
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

        dataset = CellPaintingImg(
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
            crop_size=crop_size,
            processor=processor
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
        dataset = CellPaintingImg(
            sample_index_file,
            mole_struc,
            context_length=context_length,
            subset=subset,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            crop_size=crop_size,
            processor=processor
        )
    elif args.model_type in ["cloome_phenom1", "molphenix"]:
        dataset = CellPaintingImg(
            sample_index_file,
            mole_struc,
            subset=subset,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            crop_size=crop_size,
            processor=processor
        )
    elif args.model_type in ["mae", "vit"]:
        preprocess_fn = _transform(
            args.image_resolution_train, args.image_resolution_val, is_train, None, "crop"
        )

        dataset = CellPaintingImg(
            sample_index_file,
            mole_struc,
            transforms=preprocess_fn,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            subset=subset,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            crop_size=crop_size,
            processor=processor
        )
    else:
        dataset = CellPaintingImg(
            sample_index_file,
            mole_struc,
            transforms=preprocess_fn,
            image_directory_path=img_dir,
            molecule_path=args.molecule_path,
            subset=subset,
            unique=args.unique,
            dataset=args.dataset,
            return_control=args.return_control,
            crop_size=crop_size,
            processor=processor
        )

    # num_workers = (
    #     4 * torch.cuda.device_count()
    #     if torch.get_num_threads() >= 4
    #     else torch.get_num_threads()
    # )
    num_workers=0
    # Calculate the exact number of full batches per process for distributed training.

    if num_processes > 1:

        adjusted_batch_size = int(args.batch_size / num_processes) * num_processes
        max_length = int(len(dataset) // adjusted_batch_size)
        num_samples = max_length * adjusted_batch_size

        subset_dataset = Subset(dataset, indices=range(num_samples))
    else:
        subset_dataset = dataset
        num_samples = len(dataset)

    if args.model_type == "mil_cell_clip":
        dataloader = DataLoader(
            subset_dataset,
            batch_size=int(args.batch_size / num_processes),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            subset_dataset,
            batch_size=int(args.batch_size / num_processes),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=is_train,
            collate_fn=collate_fn
        )

    dataloader.num_samples = num_samples
    dataloader.label_map = dataset.label_map

    return dataloader

def collate_fn(batch):
    """
    Adaptive collate function that handles variable batch elements.
    batch: list, each element may be:
        - ( (img_list, channel_info), mol )              # return_control=False
        - ( (img_list, channel_info), mol, control_list ) # return_control=True
    """
    batch_size = len(batch)
    

    has_control = len(batch[0]) == 3
    
    if has_control:

        image_tuples, molecules, controls = zip(*batch)
    else:

        image_tuples, molecules = zip(*batch)
        controls = [None] * batch_size
    

    all_crops_per_sample = []
    
    for img_tuple, _, _ in zip(image_tuples, molecules, controls):
        img_list, channel_info = img_tuple
        

        if not isinstance(img_list, list):
            img_list = [img_list]
        

        img_tensors = []
        for img in img_list:
            if isinstance(img, np.ndarray):
                img_tensors.append(torch.from_numpy(img))
            elif torch.is_tensor(img):
                img_tensors.append(img)
            else:

                img_tensors.append(torch.zeros(5, 3, 224, 224))
        
        if img_tensors:

            sample_crops = torch.stack(img_tensors)
        else:

            sample_crops = torch.zeros(0, 5, 3, 224, 224)
        
        all_crops_per_sample.append(sample_crops)
    

    input_ids = torch.stack([mol["input_ids"] for mol in molecules])
    attention_mask = torch.stack([mol["attention_mask"] for mol in molecules])
    
    mol_batch = {
        "input_ids": input_ids,  # (B, context_length)
        "attention_mask": attention_mask,  # (B, context_length)
    }
    

    control_crops_per_sample = None
    if has_control:
        control_crops_per_sample = []
        for ctrl in controls:
            if ctrl is None:

                if all_crops_per_sample and len(all_crops_per_sample) > 0:
                    num_crops = all_crops_per_sample[0].shape[0] if all_crops_per_sample[0].shape[0] > 0 else 5
                    control_crops = torch.zeros(num_crops, 5, 3, 224, 224)
                else:
                    control_crops = torch.zeros(5, 5, 3, 224, 224)
            elif isinstance(ctrl, list):
                if ctrl:
                    ctrl_tensors = []
                    for c in ctrl:
                        if isinstance(c, np.ndarray):
                            ctrl_tensors.append(torch.from_numpy(c))
                        elif torch.is_tensor(c):
                            ctrl_tensors.append(c)
                        else:
                            ctrl_tensors.append(torch.zeros(5, 3, 224, 224))
                    control_crops = torch.stack(ctrl_tensors)
                else:
                    control_crops = torch.zeros(0, 5, 3, 224, 224)
            elif torch.is_tensor(ctrl):
                if ctrl.dim() == 4:
                    control_crops = ctrl.unsqueeze(0)
                else:
                    control_crops = ctrl
            else:
                control_crops = torch.zeros(5, 5, 3, 224, 224)
            
            control_crops_per_sample.append(control_crops)
    

    channel_info = image_tuples[0][1]["channels"] if batch_size > 0 else None
    

    if has_control:
        return (
            (
                all_crops_per_sample,
                {"channels": channel_info},
            ),
            mol_batch,
            control_crops_per_sample
        )
    else:
        return (
            (
                all_crops_per_sample,
                {"channels": channel_info},
            ),
            mol_batch
        )


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

if __name__ == "__main__":
    args = parse_args()
    main(args)