"""
Main script for training treatment-images pair for cell painting

Mainly adopted from open_clip[1]

[1] https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/main.py
[2] https://amsword.medium.com/gradient-backpropagation-with-torch_distributed
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
import pandas as pd
import numpy as np
from scipy import stats
import warnings

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from vendor.hopfieldlayers.hflayers import Hopfield
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
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
from src.datasets import get_cellpainting_dataset
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

torch.backends.cuda.matmul.allow_tf32 = True


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
        "--seeds",
        nargs='+',
        type=int,
        # default=[42, 123, 456, 789, 3407, 2024, 999, 0, 1, 777],
        # default=[42, 123, 999, 0, 1, 777],
        default=[42, 256, 999, 3407, 1, 114514],
        help="random seeds for model training (multiple seeds supported)",
    )
    parser.add_argument(
        "--metrics_to_track",
        nargs='+',
        type=str,
        default=[
            'image_to_text_R@1', 'text_to_image_R@1', 
            'image_to_text_R@5', 'text_to_image_R@5',
            'image_to_text_R@10', 'text_to_image_R@10',
            'image_to_text_Top1%_Recall', 'text_to_image_Top1%_Recall',
            'image_to_text_Top5%_Recall', 'text_to_image_Top5%_Recall',
            'image_to_text_Top10%_Recall', 'text_to_image_Top10%_Recall',
            'image_to_text_NDCG@10', 'text_to_image_NDCG@10',
            'image_to_text_mAP', 'text_to_image_mAP',
            'zero_shot_top1_accuracy', 'val_loss'
        ],
        help="Metrics to track for best model selection",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

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
        "--is_test",
        help="whether to use training index",
        action="store_true",
        default=False,
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
        "--img_encoder",
        type=str,
        default="dinov2",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--model_card",
        type=str,
        default="facebook/dinov2-giant",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        default="google-bert/bert-base-cased",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--text_model_type",
        type=str,
        default="google",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--cross_atten",
        type=bool,
        default=True,
        help=("use cross attention or not"),
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
        default="path/to/your/data/bray2017/img_emb/dinov2-giant_ind_dataset_all.h5",
        help=("Path to training input directory."),
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        help="Dimension of input image emebddings.",
        default=1536,
    )
    parser.add_argument(
        "--mole_struc",
        type=str,
        help="mol struc format",
        default="text",
        choices=["text","morgan","smiles"]
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
        "--use_true_control",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="unseen",
        help=("dataset mode, e.g. unseen or seen"),
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="U2SO",
        help=("dataset cell type, e.g. bray2017 or jumpcp"),
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
        "--sbatch",
        help="whether to save to sbatch logs.",
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
        default=512,
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


def save_metrics_for_all_epochs(seed_outdir, all_metrics_history, epoch_numbers):
    """Save all metrics for each epoch"""
    metrics_history_path = os.path.join(seed_outdir, "all_epochs_metrics.json")
    

    epoch_metrics = {}
    for i, (epoch, metrics) in enumerate(zip(epoch_numbers, all_metrics_history)):
        epoch_metrics[epoch] = metrics
    
    with open(metrics_history_path, 'w') as f:
        json.dump(epoch_metrics, f, indent=2)
    

    csv_data = []
    for epoch, metrics in epoch_metrics.items():
        row = {'epoch': epoch}
        row.update(metrics)
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(seed_outdir, "all_epochs_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    return epoch_metrics


# def save_best_models_per_metric(model, optimizer, scheduler, epoch, metrics, seed_outdir,
#                                 metric_name, accelerator):

#     model_dir = os.path.join(seed_outdir, "best_models")
#     os.makedirs(model_dir, exist_ok=True)
#
#     model_path = os.path.join(model_dir, f"best_{metric_name}_epoch_{epoch}.pt")
#     torch.save(
#         {
#             "model": accelerator.get_state_dict(model),
#             "optimizer": optimizer.state_dict(),
#             "scheduler": scheduler.state_dict(),
#             "epoch": epoch,
#             "metric_name": metric_name,
#             "metric_value": metrics[metric_name],
#             "all_metrics": metrics,

#         },
#         model_path,
#     )
#
#     return model_path


def save_best_models_per_metric(model, optimizer, scheduler, epoch, metrics, seed_outdir,
                                metric_name, accelerator, current_best_paths):
    """Save best model for each metric, remove old best models"""
    model_dir = os.path.join(seed_outdir, "best_models")
    os.makedirs(model_dir, exist_ok=True)


    safe_metric_name = metric_name.replace('/', '_').replace(':', '_').replace('@', '_')
    filename = f"best_{safe_metric_name}.pt"
    model_path = os.path.join(model_dir, filename)


    if metric_name in current_best_paths and os.path.exists(current_best_paths[metric_name]):
        try:
            os.remove(current_best_paths[metric_name])
        except Exception as e:
            pass

    try:

        model_state = {
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": metrics[metric_name],
            "all_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            "seed": seed_outdir.split('_')[-1],
        }


        model_state["model_state_dict"] = accelerator.get_state_dict(model)


        if epoch == 1 or metric_name not in current_best_paths:
            model_state["optimizer_state_dict"] = optimizer.state_dict()
            model_state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(model_state, model_path)


        current_best_paths[metric_name] = model_path

        return model_path
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Saving model {metric_name} error: {e}")
        return None


def train_single_seed(args, seed, seed_outdir):
    """Train a single random seed"""
    try:
        set_seed(seed)

        # accelerator = Accelerator(
        #     step_scheduler_with_optimizer=False,
        # )
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1,

        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if accelerator.is_main_process:
            logger = setup_logging(seed_outdir, f'training_seed_{seed}')
            logger.info("=" * 80)
            logger.info(f"Training with seed: {seed}")
            logger.info("=" * 80)
        else:
            logger = None


        if args.wandb and accelerator.is_main_process:
            try:
                import wandb
                wandb.init(
                    project=f"Cell Painting {args.dataset}-{args.model_type}",
                    name=f"{args.model_type}-split_{args.split}-seed_{seed}",
                    config=vars(args),
                    id=f"{args.wandb_id}_{seed}" if args.wandb_id else None,
                    resume="allow",
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping wandb logging")


        train_dataloader = get_cellpainting_dataset(
            args,
            accelerator.num_processes,
            is_train=args.is_train,
            is_test=args.is_test,
        )
        eval_dataloader = get_cellpainting_dataset(
            args,
            accelerator.num_processes,
            is_train=False,
            subset=args.val_subset_ratio,
            is_test=args.is_test,
        )

        if accelerator.is_main_process:
            logger.info(
                f"Initialize training and eval loader. Number of samples: "
                f"train:{train_dataloader.num_samples}, eval:{eval_dataloader.num_samples}"
            )


        model = load_model(
            args.model_type,
            args.pretrained,
            args.image_resolution_train,
            vision_width=args.input_dim,
            loss_type=args.loss_type,
            model_card=args.model_card,
            mole_struc=args.mole_struc,
            text_model_path=args.text_model_path
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


        all_epoch_metrics = []
        all_epoch_numbers = []


        best_trackers = {}
        current_best_model_paths = {}
        for metric_name in args.metrics_to_track:
            best_trackers[metric_name] = {
                'value': float('inf') if 'loss' in metric_name else -float('inf'),
                'epoch': 0,
                'value_type': 'min' if 'loss' in metric_name else 'max',
                'model_saved': False,
                'metrics': None
            }
            current_best_model_paths[metric_name] = "None"


        if args.fine_tune_ckpt and os.path.exists(args.fine_tune_ckpt):
            ckpt = torch.load(args.fine_tune_ckpt, map_location="cpu")
            if accelerator.is_main_process:
                logger.info(f"Loading pretrained checkpoint at {args.fine_tune_ckpt}")
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "epoch" in ckpt:
                epoch = ckpt["epoch"]
            if "total_steps_done" in ckpt:
                total_steps_done = ckpt["total_steps_done"]
            if "total_training_time" in ckpt:
                total_training_time = ckpt["total_training_time"]


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

            epoch_progress = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Seed {seed} - Epoch {current_epoch + 1}/{args.epochs}",
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

                images = images.to(accelerator.device)

                if controls is not None:
                    if isinstance(controls, tuple):
                        controls = tuple(
                            item.to(accelerator.device) if hasattr(item, 'to') else item
                            for item in controls
                        )
                    elif hasattr(controls, 'to'):
                        controls = controls.to(accelerator.device)
                    elif isinstance(controls, np.ndarray):
                        controls = torch.from_numpy(controls).to(accelerator.device)

                m = model.module if accelerator.use_distributed else model


                if args.model_type in ["molphenix", "cloome_mpnn"]:
                    treatments = model.encode_mols(treatments.to(accelerator.device))
                elif args.model_type in [
                    "bert_clip",
                    "clip_channelvit",
                    "cell_clip_mae",
                    "cell_clip",
                    "pheno_cell",
                    "mil_cell_clip",
                ]:
                    if args.mole_struc=="text":
                        treatments = {k: v.to(accelerator.device) for k, v in treatments.items()}
                    else:
                        treatments=treatments
                else:
                    treatments = treatments.to(accelerator.device)

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
                total_steps_done += 1


                if (batch_idx + 1) % args.log_freq == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_loss = np.mean(train_losses[-args.log_freq:]) if len(train_losses) >= args.log_freq else train_losses[-1]
                    epoch_progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'temp': f'{m.logit_scale.data.exp():.2f}'
                    })


            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(train_losses)
            total_training_time += epoch_time

            if accelerator.is_main_process:
                logger.info(f"\nSeed {seed} | Epoch {current_epoch + 1} Summary:")
                logger.info(f"  Training Loss: {avg_epoch_loss:.4f}")
                logger.info(f"  Epoch Time: {epoch_time:.2f}s")
                logger.info(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")
                logger.info(f"  Temperature: {m.logit_scale.data.exp():.4f}")


            if (current_epoch + 1) % args.ckpt_freq == 0 and accelerator.is_main_process:
                # checkpoint_path = os.path.join(seed_outdir, f"ckpt_epoch_{current_epoch + 1:04d}.pt")
                # torch.save(
                #     {
                #         "model": accelerator.get_state_dict(model),
                #         "optimizer": optimizer.state_dict(),
                #         "scheduler": scheduler.state_dict(),
                #         "steps": total_steps_done,
                #         "epoch": current_epoch + 1,
                #         "total_training_time": total_training_time,
                #         "train_loss": avg_epoch_loss,
                #     },
                #     checkpoint_path,
                # )
                # logger.info(f"Checkpoint saved: {checkpoint_path}")
                pass


            if (current_epoch + 1) % args.eval_freq == 0 or current_epoch == 0 or current_epoch == args.epochs - 1:
                model.eval()

                if accelerator.is_main_process:
                    logger.info(f"\nEvaluating Seed {seed} at Epoch {current_epoch + 1}...")

                with torch.no_grad():
                    all_eval_image_features = []
                    all_eval_text_features = []
                    all_images = []
                    val_losses = []

                    eval_progress = tqdm(
                        enumerate(eval_dataloader),
                        total=len(eval_dataloader),
                        desc=f"Seed {seed} - Validation Epoch {current_epoch + 1}",
                        disable=not accelerator.is_main_process,
                    )

                    for eval_batch_idx, batch in eval_progress:
                        if args.return_control:
                            (images, extra_tokens), treatments, controls = batch
                        else:
                            (images, extra_tokens), treatments = batch
                            controls = None

                        images = images.to(accelerator.device)

                        if controls is not None:
                            if isinstance(controls, tuple):
                                controls = tuple(
                                    item.to(accelerator.device) if hasattr(item, 'to') else item
                                    for item in controls
                                )
                            elif hasattr(controls, 'to'):
                                controls = controls.to(accelerator.device)
                            elif isinstance(controls, np.ndarray):
                                controls = torch.from_numpy(controls).to(accelerator.device)

                        m = model.module if accelerator.use_distributed else model

                        if args.model_type in ["molphenix", "cloome_mpnn"]:
                            text_features = model.encode_mols(treatments.to(accelerator.device))
                        elif args.model_type in [
                            "bert_clip",
                            "clip_channelvit",
                            "cell_clip_mae",
                            "cell_clip",
                            "pheno_cell",
                            "mil_cell_clip",
                        ]:
                            if args.mole_struc=="text":
                                treatments = {k: v.to(accelerator.device) for k, v in treatments.items()}
                            else:
                                treatments=treatments
                        else:
                            treatments = treatments.to(accelerator.device)

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


                    if accelerator.is_main_process:
                        metrics = get_metrics(
                            all_eval_image_features, all_eval_text_features
                        )


                        avg_val_loss = np.mean(val_losses)
                        metrics['val_loss'] = avg_val_loss
                        metrics['train_loss'] = avg_epoch_loss
                        metrics['epoch'] = current_epoch + 1


                        all_epoch_metrics.append(metrics.copy())
                        all_epoch_numbers.append(current_epoch + 1)


                        logger.info(f"\nSeed {seed} | Evaluation Results - Epoch {current_epoch + 1}:")
                        logger.info(f"  Validation Loss: {avg_val_loss:.6f}")
                        logger.info(f"  Training Loss: {avg_epoch_loss:.6f}")


                        for metric_name in args.metrics_to_track:
                            if metric_name in metrics:
                                logger.info(f"  {metric_name}: {metrics[metric_name]:.6f}")


                        for metric_name in args.metrics_to_track:
                            if metric_name in metrics:
                                current_value = metrics[metric_name]
                                tracker = best_trackers[metric_name]

                                is_better = False
                                if tracker['value_type'] == 'min':
                                    is_better = current_value < tracker['value']
                                else:
                                    is_better = current_value > tracker['value']

                                if is_better:
                                    tracker['value'] = current_value
                                    tracker['epoch'] = current_epoch + 1
                                    tracker['metrics'] = metrics.copy()


                                    model_path = save_best_models_per_metric(
                                        model, optimizer, scheduler,
                                        current_epoch + 1, metrics, seed_outdir,
                                        metric_name, accelerator, current_best_model_paths
                                    )
                                    tracker['model_path']=model_path


                                    logger.info(f"✓ New best {metric_name}: {current_value:.6f} (epoch {current_epoch + 1})")


        if accelerator.is_main_process:
            logger.info("\n" + "="*80)
            logger.info(f"SEED {seed} TRAINING COMPLETED")
            logger.info("="*80)


            epoch_metrics = save_metrics_for_all_epochs(seed_outdir, all_epoch_metrics, all_epoch_numbers)


            seed_summary = {
                'seed': seed,
                'total_training_time': total_training_time,
                'total_steps': total_steps_done,
                'total_epochs': args.epochs,
                'best_performance': {},
                'final_metrics': all_epoch_metrics[-1] if all_epoch_metrics else {},
                'config': {k: v for k, v in vars(args).items() if k != 'seeds'}
            }


            for metric_name, tracker in best_trackers.items():
                if tracker['metrics'] is not None:
                    seed_summary['best_performance'][metric_name] = {
                        'value': tracker['value'],
                        'epoch': tracker['epoch'],
                        'model_path': tracker['model_path'],
                        'all_metrics_at_best': tracker['metrics']
                    }


            seed_summary_path = os.path.join(seed_outdir, "seed_summary.json")
            with open(seed_summary_path, 'w') as f:
                json.dump(seed_summary, f, indent=2)
            logger.info(f"Seed summary saved to {seed_summary_path}")


            final_model_path = os.path.join(seed_outdir, "final_model.pt")
            torch.save(
                {
                    "model": accelerator.get_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": args.epochs,
                    "total_training_time": total_training_time,
                    "seed": seed,
                    "best_performance": seed_summary['best_performance']
                },
                final_model_path,
            )
            logger.info(f"Final model saved to {final_model_path}")


            logger.info("\n" + "="*80)
            logger.info(f"SEED {seed} BEST MODEL PERFORMANCE SUMMARY")
            logger.info("="*80)

            for metric_name, tracker in best_trackers.items():
                if tracker['metrics'] is not None:
                    logger.info(f"Best {metric_name}: {tracker['value']:.6f} (epoch {tracker['epoch']})")
                    logger.info(f"  Model saved at: {tracker['model_path']}")

            logger.info("\n" + "="*80)
            logger.info(f"Seed {seed} Training Statistics:")
            logger.info(f"Total Training Time: {total_training_time:.2f}s")
            logger.info(f"Total Steps: {total_steps_done}")
            logger.info(f"Total Epochs: {args.epochs}")
            logger.info(f"Final Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            logger.info("="*80)


        seed_results = {
            'seed': seed,
            'total_training_time': total_training_time,
            'total_steps': total_steps_done,
            'best_performance': best_trackers,
            'all_epoch_metrics': all_epoch_metrics,
            'all_epoch_numbers': all_epoch_numbers,
            'seed_summary': seed_summary if accelerator.is_main_process else None
        }

        return seed_results

    finally:

        if torch.cuda.is_available():
            try:

                if 'model' in locals():
                    del model
                if 'optimizer' in locals():
                    del optimizer
                if 'scheduler' in locals():
                    del scheduler
                if 'train_dataloader' in locals():
                    del train_dataloader
                if 'eval_dataloader' in locals():
                    del eval_dataloader


                torch.cuda.empty_cache()


                import gc
                gc.collect()


                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Cleanup memory error: {e}")


def aggregate_seed_results(all_results, output_dir, args):
    """Aggregate results across all random seeds"""
    os.makedirs(output_dir, exist_ok=True)
    

    summary_logger = setup_logging(output_dir, 'multi_seed_summary')
    summary_logger.info("\n" + "="*80)
    summary_logger.info("MULTI-SEED TRAINING SUMMARY")
    summary_logger.info("="*80)
    

    all_best_metrics_data = []
    all_metrics_detailed = []
    
    for result in all_results:
        seed = result['seed']
        best_performance = result['best_performance']
        

        seed_best_metrics = {'seed': seed}
        for metric_name, tracker in best_performance.items():
            if tracker['metrics'] is not None:
                seed_best_metrics[metric_name] = tracker['value']
                

                metrics_at_best = tracker['metrics']
                metrics_at_best['seed'] = seed
                metrics_at_best['metric_name'] = metric_name
                metrics_at_best['epoch'] = tracker['epoch']
                all_metrics_detailed.append(metrics_at_best)
        
        all_best_metrics_data.append(seed_best_metrics)
    

    df_best_results = pd.DataFrame(all_best_metrics_data)
    

    df_detailed_metrics = pd.DataFrame(all_metrics_detailed)
    detailed_metrics_path = os.path.join(output_dir, "all_seeds_detailed_metrics.csv")
    df_detailed_metrics.to_csv(detailed_metrics_path, index=False)
    

    summary_stats = {}
    all_metrics_stats = {}
    

    for metric_name in args.metrics_to_track:
        if metric_name in df_best_results.columns:
            values = df_best_results[metric_name].dropna()
            if len(values) > 0:
                metric_stats = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'range': f"{values.min():.6f} - {values.max():.6f}",
                    'cv': float(values.std() / values.mean() if values.mean() != 0 else 0),
                    'num_seeds': len(values),
                    'values': values.tolist(),
                    'best_seed': int(values.idxmin() if 'loss' in metric_name else values.idxmax()),
                    'best_value': float(values.min() if 'loss' in metric_name else values.max())
                }
                summary_stats[metric_name] = metric_stats
    

    all_available_metrics = []
    for col in df_detailed_metrics.columns:
        if col not in ['seed', 'metric_name', 'epoch', 'train_loss', 'val_loss'] and col in args.metrics_to_track:
            all_available_metrics.append(col)
    
    for metric_name in all_available_metrics:
        if metric_name in df_detailed_metrics.columns:
            values = df_detailed_metrics[metric_name].dropna()
            if len(values) > 0:
                all_metrics_stats[metric_name] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'num_values': len(values)
                }
    

    stats_path = os.path.join(output_dir, "all_metrics_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'tracked_metrics_summary': summary_stats,
            'all_metrics_statistics': all_metrics_stats,
            'config': {k: v for k, v in vars(args).items() if k != 'seeds'}
        }, f, indent=2)
    

    generate_comprehensive_report(df_best_results, df_detailed_metrics, 
                                 summary_stats, all_metrics_stats, output_dir, args)
    
    return df_best_results, summary_stats


def generate_comprehensive_report(df_best_results, df_detailed_metrics, 
                                 summary_stats, all_metrics_stats, output_dir, args):
    """Generate comprehensive report"""
    report_path = os.path.join(output_dir, "comprehensive_statistical_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Multi-Seed Training Comprehensive Statistical Report\n")
        f.write("="*80 + "\n\n")
        

        f.write("1. Experiment Overview\n")
        f.write("-"*40 + "\n")
        f.write(f"Number of seeds: {len(df_best_results)}\n")
        f.write(f"Tracked metrics: {', '.join(args.metrics_to_track)}\n")
        f.write(f"Total evaluations: {len(df_detailed_metrics)}\n")
        f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        

        f.write("2. Best performance per seed\n")
        f.write("-"*40 + "\n")
        for idx, row in df_best_results.iterrows():
            f.write(f"\nSeed {row['seed']}:\n")
            for metric_name in args.metrics_to_track:
                if metric_name in row and pd.notna(row[metric_name]):
                    f.write(f"  Best {metric_name}: {row[metric_name]:.6f}\n")
        

        f.write("\n3. Tracked metricsStatistical Summary\n")
        f.write("-"*40 + "\n")
        

        metric_categories = {
            'Basic retrieval metrics': [m for m in args.metrics_to_track if 'R@' in m],
            'Ranking quality metrics': [m for m in args.metrics_to_track if 'NDCG' in m or 'mAP' in m or 'P@' in m],
            'Zero-shot classification metrics': [m for m in args.metrics_to_track if 'accuracy' in m],
            'Loss metrics': [m for m in args.metrics_to_track if 'loss' in m]
        }
        
        for category_name, metrics_in_category in metric_categories.items():
            available_metrics = [m for m in metrics_in_category if m in summary_stats]
            if available_metrics:
                f.write(f"\n{category_name}:\n")
                for metric_name in available_metrics:
                    stats_info = summary_stats[metric_name]
                    f.write(f"\n  {metric_name}:\n")
                    f.write(f"    Mean: {stats_info['mean']:.6f} ± {stats_info['std']:.6f}\n")
                    f.write(f"    Range: {stats_info['range']}\n")
                    f.write(f"    Median: {stats_info['median']:.6f}\n")
                    f.write(f"    CV: {stats_info['cv']:.4f}\n")
                    f.write(f"    Best seed: {stats_info['best_seed']} (value: {stats_info['best_value']:.6f})\n")
        

        f.write("\n4. Stability Analysis\n")
        f.write("-"*40 + "\n")
        
        stability_ranking = []
        for metric_name, stats_info in summary_stats.items():
            if stats_info['cv'] > 0:
                stability_ranking.append((metric_name, stats_info['cv']))
        
        if stability_ranking:
            f.write("\nMetric stability ranking (smaller CV = more stable):\n")
            stability_ranking.sort(key=lambda x: x[1])
            for metric_name, cv in stability_ranking:
                f.write(f"  {metric_name}: {cv:.4f}\n")
        

        f.write("\n5. Inter-metric Correlation Analysis\n")
        f.write("-"*40 + "\n")
        

        numeric_metrics = []
        for metric_name in args.metrics_to_track:
            if metric_name in df_best_results.columns:
                if df_best_results[metric_name].dtype in [np.float64, np.float32, np.int64]:
                    numeric_metrics.append(metric_name)
        
        if len(numeric_metrics) >= 2:
            correlation_matrix = df_best_results[numeric_metrics].corr()
            f.write("\nPearson correlation matrix between metrics:\n")
            

            high_corr_threshold = 0.7
            for i, metric1 in enumerate(numeric_metrics):
                for j, metric2 in enumerate(numeric_metrics):
                    if i < j:
                        corr = correlation_matrix.loc[metric1, metric2]
                        if abs(corr) > high_corr_threshold:
                            f.write(f"  {metric1} ↔ {metric2}: {corr:.3f}\n")
        

        f.write("\n6. Recommendations and Conclusions\n")
        f.write("-"*40 + "\n")
        
        f.write("\nBased on multi-seed experiments, recommendations:\n")
        

        if stability_ranking:
            most_stable = stability_ranking[0][0]
            least_stable = stability_ranking[-1][0]
            f.write(f"1. The most stable metric is {most_stable} (CV={stability_ranking[0][1]:.4f})\n")
            f.write(f"2. The least stable metric is {least_stable} (CV={stability_ranking[-1][1]:.4f})\n")
        

        best_seed_by_metric = {}
        for metric_name, stats_info in summary_stats.items():
            if 'best_seed' in stats_info:
                best_seed_by_metric[metric_name] = stats_info['best_seed']
        
        if best_seed_by_metric:
            seed_counts = {}
            for seed in best_seed_by_metric.values():
                seed_counts[seed] = seed_counts.get(seed, 0) + 1
            
            if seed_counts:
                most_common_seed = max(seed_counts.items(), key=lambda x: x[1])
                f.write(f"3. Seed {most_common_seed[0]}  on  {most_common_seed[1]}  metrics \n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("End of report\n")
        f.write("="*80 + "\n")
    
    print(f"Comprehensive statistical report saved to: {report_path}")
    

    # html_report_path = os.path.join(output_dir, "statistical_summary.html")
    # generate_html_report(df_best_results, summary_stats, html_report_path, args)


def generate_html_report(df_best_results, summary_stats, output_path, args):
    """Generate HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Seed Training Statistical Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric-card {{ 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                background-color: #f9f9f9;
            }}
            .best-seed {{ color: green; font-weight: bold; }}
            .warning {{ color: orange; font-weight: bold; }}
            .summary-box {{ 
                background-color: #e8f4f8; 
                padding: 15px; 
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Multi-Seed Training Statistical Report</h1>
        
        <div class="summary-box">
            <h2>Experiment Overview</h2>
            <p><strong>Number of seeds:</strong> {len(df_best_results)}</p>
            <p><strong>Generated at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Best performance per seed</h2>
        {df_best_results.to_html(index=False, float_format=lambda x: f'{x:.6f}')}
        
        <h2>MetricStatistical Summary</h2>
    """
    

    metric_categories = {
        'Basic retrieval metrics': [m for m in args.metrics_to_track if 'R@' in m],
        'Ranking quality metrics': [m for m in args.metrics_to_track if 'NDCG' in m or 'mAP' in m or 'P@' in m],
        'Zero-shot classification metrics': [m for m in args.metrics_to_track if 'accuracy' in m],
        'Loss metrics': [m for m in args.metrics_to_track if 'loss' in m]
    }
    
    for category_name, metrics_in_category in metric_categories.items():
        available_metrics = [m for m in metrics_in_category if m in summary_stats]
        if available_metrics:
            html_content += f'<h3>{category_name}</h3>'
            
            for metric_name in available_metrics:
                stats_info = summary_stats[metric_name]
                is_loss = 'loss' in metric_name
                
                html_content += f"""
                <div class="metric-card">
                    <h4>{metric_name}</h4>
                    <p><strong>Mean:</strong> {stats_info['mean']:.6f} ± {stats_info['std']:.6f}</p>
                    <p><strong>Range:</strong> {stats_info['range']}</p>
                    <p><strong>Median:</strong> {stats_info['median']:.6f}</p>
                    <p><strong>CV:</strong> {stats_info['cv']:.4f}</p>
                    <p><strong>Best seed:</strong> <span class="best-seed">{stats_info['best_seed']}</span> (value: {stats_info['best_value']:.6f})</p>
                </div>
                """
    
    html_content += """
        <h2>Stability Analysis</h2>
        <table>
            <tr><th>Metric</th><th>CV(CV)</th><th>Stability Rating</th></tr>
    """
    

    for metric_name, stats_info in summary_stats.items():
        cv = stats_info['cv']
        if cv < 0.1:
            stability = "Very Stable"
            color = "green"
        elif cv < 0.2:
            stability = "Stable"
            color = "lightgreen"
        elif cv < 0.3:
            stability = "Moderate"
            color = "orange"
        else:
            stability = "Unstable"
            color = "red"
        
        html_content += f"""
        <tr>
            <td>{metric_name}</td>
            <td>{cv:.4f}</td>
            <td style="color: {color}">{stability}</td>
        </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Recommendations</h2>
        <ul>
            <li>For the most stable metrics, consider reducing the number of random seeds to save computational resources</li>
            <li>For unstable metrics, more random seeds may be needed for reliable results</li>
            <li>Choose seeds that perform well on multiple metrics as the final model</li>
        </ul>
        
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")


def main(args):
    """Main training function"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    base_outdir = os.path.join(
        constants.SBATCH_OUT_DIR if args.sbatch else constants.OUT_DIR,
        "results",
        args.dataset,
        args.dataset_mode,
        args.cell_type,
        "models",
        args.model_type,
        f"{args.model_type}_img{args.img_encoder}_molstru{args.mole_struc}_loss{args.loss_type}_batchsize{args.batch_size}_epochs{args.epochs}_rs{len(args.seeds)}_topk_{timestamp}"
    )
    
    args.model_outdir = base_outdir
    os.makedirs(base_outdir, exist_ok=True)
    

    config_path = os.path.join(base_outdir, "training_config.json")
    with open(config_path, 'w') as f:
        config_dict = vars(args)
        json.dump(config_dict, f, indent=2)
    

    main_logger = setup_logging(base_outdir, 'multi_seed_training')
    main_logger.info("=" * 80)
    main_logger.info(f"MULTI-SEED TRAINING STARTED")
    main_logger.info("=" * 80)
    main_logger.info(f"Training Configuration")
    main_logger.info("=" * 80)
    for arg in vars(args):
        main_logger.info(f"{arg}: {getattr(args, arg)}")
    main_logger.info("=" * 80)
    main_logger.info(f"Number of GPU available: {torch.cuda.device_count()}")

    

    all_results = []
    
    for seed_idx, seed in enumerate(args.seeds):
        main_logger.info(f"\n{'='*60}")
        main_logger.info(f"Starting training for seed {seed} ({seed_idx + 1}/{len(args.seeds)})")
        main_logger.info(f"{'='*60}")
        
        seed_outdir = os.path.join(base_outdir, f"seed_{seed}")
        seed_start_time = time.time()
        
        seed_results = train_single_seed(args, seed, seed_outdir)
        seed_time = time.time() - seed_start_time
        
        seed_results['training_time'] = seed_time
        all_results.append(seed_results)
        
        main_logger.info(f"\nSeed {seed} completed in {seed_time:.2f}s")


        if torch.cuda.is_available():
            main_logger.info("Clearing GPU memory for next seed......")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()


            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            main_logger.info(f"After cleanup GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    

    main_logger.info("\n" + "="*80)
    main_logger.info("AGGREGATING RESULTS FROM ALL SEEDS")
    main_logger.info("="*80)
    
    df_results, summary_stats = aggregate_seed_results(all_results, base_outdir, args)
    

    multi_seed_results = {
        'all_results': all_results,
        'summary_stats': summary_stats,
        'config': vars(args)
    }
    
    results_path = os.path.join(base_outdir, "multi_seed_complete_results.json")
    with open(results_path, 'w') as f:
        json.dump(multi_seed_results, f, indent=2, default=str)
    
    main_logger.info(f"\nMulti-seed results saved to: {results_path}")
    

    total_training_time = sum(r['training_time'] for r in all_results)
    main_logger.info("\n" + "="*80)
    main_logger.info("MULTI-SEED TRAINING COMPLETED")
    main_logger.info("="*80)
    main_logger.info(f"Total training time: {total_training_time:.2f}s")
    main_logger.info(f"Average time per seed: {total_training_time/len(args.seeds):.2f}s")
    main_logger.info(f"All results saved in: {base_outdir}")
    main_logger.info("="*80)
    
    return all_results


if __name__ == "__main__":
    args = parse_args()
    results = main(args)