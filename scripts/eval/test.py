"""
Test script: load trained best model and compute per-sample metrics on the test set
"""

import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import sys

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import constants
from src.clip.clip import load_model
from src.datasets import get_cellpainting_dataset
from src.helpler import all_gather,get_metrics_with_actual_ids


def setup_test_logging(output_dir, name='testing'):
    """Setup test logging"""
    os.makedirs(output_dir, exist_ok=True)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    log_file = os.path.join(output_dir, 'testing.log')
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


def get_metrics_with_per_sample(image_features, text_features, sample_keys, logit_scale=None):
    """Evaluate retrieval metrics with per-sample details"""


    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)


    if logit_scale is not None:
        similarity_matrix = logit_scale * image_features_norm @ text_features_norm.t()
    else:
        similarity_matrix = image_features_norm @ text_features_norm.t()

    n_samples = len(image_features)


    if len(sample_keys) != n_samples:
        raise ValueError(f"sample_keys count({len(sample_keys)}) vs feature count({n_samples})mismatch")

    metrics = {}


    per_sample_metrics, overall_metrics = compute_detailed_retrieval_metrics_with_ids(
        image_features, text_features, similarity_matrix, sample_keys
    )
    metrics.update(overall_metrics)


    metrics["per_sample_metrics"] = per_sample_metrics


    metrics.update(compute_top_k_percent_recall(similarity_matrix))
    metrics.update(compute_ranking_quality_metrics(similarity_matrix))
    metrics.update(compute_zero_shot_classification_metrics(similarity_matrix))

    return metrics


def extract_features(model, dataloader, args, accelerator, collect_keys=True):
    """Extract features and sample keys from dataloader"""
    model.eval()

    all_image_features = []
    all_text_features = []
    all_sample_keys = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", disable=not accelerator.is_main_process):

            if args.return_control:
                (images, extra_tokens), treatments, controls, sample_key = batch
            else:
                (images, extra_tokens), treatments, sample_key = batch
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
                if args.mole_struc == "text":
                    treatments = {k: v.to(accelerator.device) for k, v in treatments.items()}
                else:
                    treatments = treatments.to(accelerator.device)
            else:
                treatments = treatments.to(accelerator.device)

            if args.model_type == "cell_clip" or args.model_type == "pheno_cell":
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


            all_image_features.append(img_features.cpu())
            all_text_features.append(text_features.cpu())


            if collect_keys:
                if isinstance(sample_key, torch.Tensor):
                    all_sample_keys.extend(sample_key.cpu().numpy().tolist())
                else:
                    all_sample_keys.extend(sample_key)


    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)


    if accelerator.use_distributed:
        all_image_features = accelerator.gather_for_metrics(all_image_features)
        all_text_features = accelerator.gather_for_metrics(all_text_features)


    if collect_keys and accelerator.is_main_process:

        if accelerator.use_distributed:

            all_sample_keys_gathered = [None] * accelerator.num_processes
            accelerator.gather_object(all_sample_keys, all_sample_keys_gathered)

            if accelerator.is_main_process:

                all_sample_keys = []
                for proc_keys in all_sample_keys_gathered:
                    if proc_keys is not None:
                        all_sample_keys.extend(proc_keys)
        else:
            all_sample_keys = all_sample_keys
    else:
        all_sample_keys = None

    return all_image_features, all_text_features, all_sample_keys


def test_model_on_dataset(model_path, args, accelerator, test_dataloader, output_dir, metric_name):
    """Test a single model on dataset"""
    logger = logging.getLogger('testing')

    logger.info(f"\nLoading model: {model_path}")


    checkpoint = torch.load(model_path, map_location="cpu",weights_only=False)


    model_config = {
        'model_type': args.model_type,
        'pretrained': args.pretrained,
        'image_resolution_train': args.image_resolution_train,
        'vision_width': args.input_dim,
        'loss_type': args.loss_type,
        'model_card': args.model_card,
        'mole_struc': args.mole_struc,
        'text_model_path': args.text_model_path
    }


    model = load_model(**model_config)


    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError("No model weights found in checkpoint")


    model = accelerator.prepare(model)
    test_dataloader = accelerator.prepare(test_dataloader)


    logger.info("Extracting test set features......")
    image_features, text_features, sample_keys = extract_features(
        model, test_dataloader, args, accelerator, collect_keys=True
    )


    if accelerator.is_main_process:
        logger.info(f"Feature extraction done, image shape: {image_features.shape}, text shape: {text_features.shape}")

        if sample_keys is None or len(sample_keys) != len(image_features):
            logger.warning(
                f"sample_keys count({len(sample_keys) if sample_keys else 0}) vs feature count({len(image_features)})mismatch")

            sample_keys = list(range(len(image_features)))


        logit_scale = None
        # if hasattr(model.module if accelerator.use_distributed else model, 'logit_scale'):
        #     m = model.module if accelerator.use_distributed else model
        #     logit_scale = m.logit_scale


        logger.info("Computing detailed metrics......")
        metrics = get_metrics_with_actual_ids(
            image_features, text_features, sample_keys, logit_scale
        )


        metrics['model_path'] = model_path
        metrics['metric_name'] = metric_name
        metrics['checkpoint_info'] = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'metric_value': checkpoint.get('metric_value', 'unknown'),
            'seed': checkpoint.get('seed', 'unknown')
        }


        model_test_dir = os.path.join(output_dir, f"best_{metric_name}")
        os.makedirs(model_test_dir, exist_ok=True)


        overall_metrics = {k: v for k, v in metrics.items() if k != "per_sample_metrics"}
        overall_metrics_path = os.path.join(model_test_dir, "overall_metrics.json")
        with open(overall_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, indent=2, ensure_ascii=False)


        per_sample_metrics = metrics["per_sample_metrics"]
        per_sample_path = os.path.join(model_test_dir, "per_sample_metrics.json")


        serializable_per_sample = {}
        for sample_key, sample_data in per_sample_metrics.items():
            serializable_sample = {}
            for key, value in sample_data.items():
                if isinstance(value, np.ndarray):
                    serializable_sample[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    serializable_sample[key] = value.cpu().numpy().tolist()
                else:
                    serializable_sample[key] = value
            serializable_per_sample[str(sample_key)] = serializable_sample

        with open(per_sample_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_per_sample, f, indent=2, ensure_ascii=False)


        summary = create_per_sample_summary(per_sample_metrics)
        summary_path = os.path.join(model_test_dir, "sample_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


        csv_data = create_sample_csv_data(per_sample_metrics)
        csv_path = os.path.join(model_test_dir, "per_sample_results.csv")
        csv_data.to_csv(csv_path, index=False)

        logger.info(f"Test results saved to: {model_test_dir}")
        logger.info(f"Overall metrics: {overall_metrics_path}")
        logger.info(f"Per-sample metrics: {per_sample_path}")
        logger.info(f"Sample summary: {summary_path}")
        logger.info(f"CSV summary: {csv_path}")


        logger.info("\nKey metrics:")
        for key in ['image_to_text_R@1', 'text_to_image_R@1',
                    'image_to_text_R@5', 'text_to_image_R@5',
                    'image_to_text_R@10', 'text_to_image_R@10']:
            if key in overall_metrics:
                logger.info(f"  {key}: {overall_metrics[key]:.4f}")

        return metrics
    else:
        return None


def create_per_sample_summary(per_sample_metrics):
    """Create per-sample summary statistics"""
    summary = {
        'total_samples': len(per_sample_metrics),
        'retrieval_success': {
            'image_to_text_top1': sum(1 for m in per_sample_metrics.values() if m["image_to_text_rank"] == 0),
            'text_to_image_top1': sum(1 for m in per_sample_metrics.values() if m["text_to_image_rank"] == 0),
            'both_top1': sum(1 for m in per_sample_metrics.values() if
                             m["image_to_text_rank"] == 0 and m["text_to_image_rank"] == 0),
        },
        'rank_distribution': {
            'image_to_text': {
                'top1': sum(1 for m in per_sample_metrics.values() if m["image_to_text_rank"] == 0),
                'top5': sum(1 for m in per_sample_metrics.values() if m["image_to_text_rank"] < 5),
                'top10': sum(1 for m in per_sample_metrics.values() if m["image_to_text_rank"] < 10),
                'top50': sum(1 for m in per_sample_metrics.values() if m["image_to_text_rank"] < 50),
                'worst_10': [],
            },
            'text_to_image': {
                'top1': sum(1 for m in per_sample_metrics.values() if m["text_to_image_rank"] == 0),
                'top5': sum(1 for m in per_sample_metrics.values() if m["text_to_image_rank"] < 5),
                'top10': sum(1 for m in per_sample_metrics.values() if m["text_to_image_rank"] < 10),
                'top50': sum(1 for m in per_sample_metrics.values() if m["text_to_image_rank"] < 50),
                'worst_10': [],
            }
        },
        'hard_samples': [],
        'easy_samples': []
    }


    all_samples = list(per_sample_metrics.items())
    all_samples.sort(key=lambda x: max(x[1]["image_to_text_rank"], x[1]["text_to_image_rank"]), reverse=True)

    for sample_key, sample_data in all_samples[:10]:
        summary['hard_samples'].append({
            'sample_key': sample_key,
            'image_to_text_rank': int(sample_data["image_to_text_rank"]),
            'text_to_image_rank': int(sample_data["text_to_image_rank"]),
            'max_rank': int(max(sample_data["image_to_text_rank"], sample_data["text_to_image_rank"]))
        })


    easy_samples = [(k, v) for k, v in per_sample_metrics.items()
                    if v["image_to_text_rank"] == 0 and v["text_to_image_rank"] == 0]

    for sample_key, sample_data in easy_samples[:10]:
        summary['easy_samples'].append({
            'sample_key': sample_key,
            'image_to_text_top_matches': sample_data.get('image_to_text_top_keys', [])[:3],
            'text_to_image_top_matches': sample_data.get('text_to_image_top_keys', [])[:3]
        })


    total = summary['total_samples']
    summary['retrieval_success_percentage'] = {
        'image_to_text_top1': summary['retrieval_success']['image_to_text_top1'] / total * 100,
        'text_to_image_top1': summary['retrieval_success']['text_to_image_top1'] / total * 100,
        'both_top1': summary['retrieval_success']['both_top1'] / total * 100,
    }

    return summary


def create_sample_csv_data(per_sample_metrics):
    """Create CSV data for per-sample results"""
    rows = []

    for sample_key, sample_data in per_sample_metrics.items():
        row = {
            'sample_key': sample_key,
            'image_to_text_rank': sample_data["image_to_text_rank"] + 1,
            'text_to_image_rank': sample_data["text_to_image_rank"] + 1,
            'is_top1_img2txt': sample_data["image_to_text_rank"] == 0,
            'is_top5_img2txt': sample_data["image_to_text_rank"] < 5,
            'is_top10_img2txt': sample_data["image_to_text_rank"] < 10,
            'is_top1_txt2img': sample_data["text_to_image_rank"] == 0,
            'is_top5_txt2img': sample_data["text_to_image_rank"] < 5,
            'is_top10_txt2img': sample_data["text_to_image_rank"] < 10,
            'image_to_text_top1_match': sample_data.get('image_to_text_top_keys', [])[0] if sample_data.get(
                'image_to_text_top_keys') else None,
            'text_to_image_top1_match': sample_data.get('text_to_image_top_keys', [])[0] if sample_data.get(
                'text_to_image_top_keys') else None,
            'max_rank': max(sample_data["image_to_text_rank"], sample_data["text_to_image_rank"]) + 1,
            'retrieval_category': get_retrieval_category(sample_data)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_retrieval_category(sample_data):
    """Categorize samples by retrieval results"""
    img_rank = sample_data["image_to_text_rank"]
    txt_rank = sample_data["text_to_image_rank"]

    if img_rank == 0 and txt_rank == 0:
        return "perfect"
    elif img_rank == 0 or txt_rank == 0:
        return "good"
    elif img_rank < 10 and txt_rank < 10:
        return "fair"
    elif img_rank < 50 or txt_rank < 50:
        return "poor"
    else:
        return "hard"


def aggregate_test_results(test_results_dir, seeds, metrics_to_track):
    """Aggregate test results across all seeds"""
    all_results = []

    for seed in seeds:
        seed_dir = os.path.join(test_results_dir, f"seed_{seed}")

        if os.path.exists(seed_dir):
            for metric_name in metrics_to_track:
                metric_dir = os.path.join(seed_dir, "test_results", f"best_{metric_name}")
                overall_metrics_path = os.path.join(metric_dir, "overall_metrics.json")

                if os.path.exists(overall_metrics_path):
                    with open(overall_metrics_path, 'r') as f:
                        metrics = json.load(f)


                    result = {
                        'seed': seed,
                        'metric_name': metric_name,
                        'image_to_text_R@1': metrics.get('image_to_text_R@1', None),
                        'text_to_image_R@1': metrics.get('text_to_image_R@1', None),
                        'image_to_text_R@5': metrics.get('image_to_text_R@5', None),
                        'text_to_image_R@5': metrics.get('text_to_image_R@5', None),
                        'image_to_text_R@10': metrics.get('image_to_text_R@10', None),
                        'text_to_image_R@10': metrics.get('text_to_image_R@10', None),
                        'model_path': metrics.get('model_path', '')
                    }

                    all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(test_results_dir, "all_seeds_test_summary.csv")
        df.to_csv(summary_path, index=False)


        stats_summary = {}
        for metric in ['image_to_text_R@1', 'text_to_image_R@1',
                       'image_to_text_R@5', 'text_to_image_R@5',
                       'image_to_image_R@10', 'text_to_image_R@10']:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    stats_summary[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median())
                    }

        stats_path = os.path.join(test_results_dir, "test_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats_summary, f, indent=2)

        return df, stats_summary
    else:
        return None, None


def parse_test_args():
    """Parse test arguments"""
    parser = argparse.ArgumentParser(description="Test trained models")


    parser.add_argument(
        "--training_results_dir",
        type=str,
        required=True,
        help="Directory containing training results for each seed"
    )


    parser.add_argument(
        "--split_label_dir",
        type=str,
        default=constants.SPLIT_LABEL_DIR,
        help="Data split label directory"
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="Dataset split index"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bray2017",
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="unseen",
        help="Dataset mode"
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="U2SO",
        help="Cell type"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=False,
        help="Image data directory"
    )
    parser.add_argument(
        "--molecule_path",
        type=str,
        required=False,
        help="Molecule data path"
    )
    parser.add_argument(
        "--mole_struc",
        type=str,
        default="text",
        choices=["text", "morgan", "smiles"],
        help="Molecule structure format"
    )
    parser.add_argument(
        "--return_control",
        action="store_true",
        default=False,
        help="Whether to return control embeddings"
    )
    parser.add_argument(
        "--is_test",
        action="store_true",
        default=True,
        help="Use test set"
    )


    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type"
    )
    parser.add_argument(
        "--img_encoder",
        type=str,
        required=False,
        help="Image encoder type"
    )
    parser.add_argument(
        "--model_card",
        type=str,
        required=False,
        help="Pretrained model card"
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        required=False,
        help="Text model path"
    )
    parser.add_argument(
        "--text_model_type",
        type=str,
        default="google",
        help="Text model type"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use pretrained model"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="clip",
        help="Loss type"
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1536,
        help="Input dimension"
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Test batch size"
    )
    parser.add_argument(
        "--image_resolution_val",
        type=int,
        default=224,
        help="Validation/test image resolution"
    )
    parser.add_argument(
        "--test_seeds",
        nargs='+',
        type=int,
        help="List of seeds to test; if not specified, test all seeds"
    )
    parser.add_argument(
        "--metrics_to_test",
        nargs='+',
        type=str,
        default=[
            'image_to_text_R@1', 'text_to_image_R@1',
            'image_to_text_R@5', 'text_to_image_R@5',
            'image_to_text_R@10', 'text_to_image_R@10',
            'image_to_text_Top1%_Recall', 'text_to_image_Top1%_Recall',
            'image_to_text_mAP', 'text_to_image_mAP'
        ],
        help="List of metrics to test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Test output directory; defaults to test_results inside training directory"
    )
    parser.add_argument(
        "--test_final_model",
        action="store_true",
        default=False,
        help="Also test the final model"
    )

    return parser.parse_args()


def main():
    """Main test function"""
    args = parse_test_args()


    if args.output_dir:
        test_output_dir = args.output_dir
    else:
        test_output_dir = os.path.join(args.training_results_dir, "test_results")

    os.makedirs(test_output_dir, exist_ok=True)


    logger = setup_test_logging(test_output_dir)
    logger.info("=" * 80)
    logger.info("Model testing started")
    logger.info("=" * 80)



    accelerator = Accelerator()


    training_config_path = os.path.join(args.training_results_dir, "training_config.json")
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)


        for key, value in training_config.items():
            # if not hasattr(args, key):
            setattr(args, key, value)
            logger.info(f"Inherited parameter from training config: {key} = {value}")


    config_path = os.path.join(test_output_dir, "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Test config saved to: {config_path}")


    logger.info("Loading test dataset......")


    args.is_train = False
    args.val_subset_ratio = 1.0

    test_dataloader = get_cellpainting_dataset(
        args,
        accelerator.num_processes,
        is_train=False,
        subset=1.0,
        is_test=True
    )

    logger.info(f"Test dataset size: {test_dataloader.num_samples}")


    if args.test_seeds:
        seeds_to_test = [int(seed) for seed in args.test_seeds]
    else:

        seeds_to_test = []
        for item in os.listdir(args.training_results_dir):
            if item.startswith("seed_"):
                try:
                    seed = int(item.split("_")[1])
                    seeds_to_test.append(seed)
                except ValueError:
                    continue

    logger.info(f"Seeds to test: {seeds_to_test}")


    all_test_results = []

    for seed in seeds_to_test:
        seed_dir = os.path.join(args.training_results_dir, f"seed_{seed}")

        if not os.path.exists(seed_dir):
            logger.warning(f"Seed {seed} directory not found: {seed_dir}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing Seed {seed}")
        logger.info(f"{'=' * 60}")


        seed_test_dir = os.path.join(test_output_dir, f"seed_{seed}")
        os.makedirs(seed_test_dir, exist_ok=True)


        for metric_name in args.metrics_to_test:

            safe_metric_name = metric_name.replace('/', '_').replace(':', '_').replace('@', '_')
            model_filename = f"best_{safe_metric_name}.pt"
            model_path = os.path.join(seed_dir, "best_models", model_filename)

            if os.path.exists(model_path):
                logger.info(f"Testing model: {metric_name}")


                metrics = test_model_on_dataset(
                    model_path, args, accelerator, test_dataloader,
                    seed_test_dir, metric_name
                )

                if metrics and accelerator.is_main_process:
                    all_test_results.append({
                        'seed': seed,
                        'metric_name': metric_name,
                        'model_path': model_path,
                        'metrics': metrics
                    })
            else:
                logger.warning(f"Model file not found: {model_path}")


        if args.test_final_model:
            final_model_path = os.path.join(seed_dir, "final_model.pt")
            if os.path.exists(final_model_path):
                logger.info(f"Test final model")

                metrics = test_model_on_dataset(
                    final_model_path, args, accelerator, test_dataloader,
                    seed_test_dir, "final_model"
                )

                if metrics and accelerator.is_main_process:
                    all_test_results.append({
                        'seed': seed,
                        'metric_name': 'final_model',
                        'model_path': final_model_path,
                        'metrics': metrics
                    })
            else:
                logger.warning(f"Final model file not found: {final_model_path}")


    if accelerator.is_main_process and all_test_results:
        logger.info("\n" + "=" * 80)
        logger.info("Aggregating all test results")
        logger.info("=" * 80)


        summary_rows = []
        for result in all_test_results:
            metrics = result['metrics']
            overall_metrics = {k: v for k, v in metrics.items() if k != "per_sample_metrics"}

            row = {
                'seed': result['seed'],
                'metric_name': result['metric_name'],
                'model_path': result['model_path']
            }


            for key in ['image_to_text_R@1', 'text_to_image_R@1',
                        'image_to_text_R@5', 'text_to_image_R@5',
                        'image_to_text_R@10', 'text_to_image_R@10',
                        'zero_shot_top1_accuracy']:
                if key in overall_metrics:
                    row[key] = overall_metrics[key]

            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(test_output_dir, "all_models_test_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Test summary saved to: {summary_path}")


        generate_test_report(all_test_results, test_output_dir, args)

    logger.info("\n" + "=" * 80)
    logger.info("Model testing completed")
    logger.info(f"Results saved to: {test_output_dir}")
    logger.info("=" * 80)


def generate_test_report(all_test_results, output_dir, args):
    """Generate detailed test report"""
    report_path = os.path.join(output_dir, "detailed_test_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Detailed Model Test Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test dataset: {args.dataset}\n")
        f.write(f"Test mode: {args.dataset_mode}\n")
        f.write(f"Cell type: {args.cell_type}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Number of test seeds: {len(set(r['seed'] for r in all_test_results))}\n")
        f.write(f"Total number of models tested: {len(all_test_results)}\n\n")


        seeds = sorted(set(r['seed'] for r in all_test_results))

        for seed in seeds:
            seed_results = [r for r in all_test_results if r['seed'] == seed]
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Seed {seed} Test results\n")
            f.write(f"{'=' * 60}\n\n")

            for result in seed_results:
                metrics = result['metrics']
                overall_metrics = {k: v for k, v in metrics.items() if k != "per_sample_metrics"}

                f.write(f"Model: {result['metric_name']}\n")
                f.write(f"Path: {result['model_path']}\n")


                f.write("Key metrics:\n")
                for key in ['image_to_text_R@1', 'text_to_image_R@1',
                            'image_to_text_R@5', 'text_to_image_R@5',
                            'image_to_text_R@10', 'text_to_image_R@10']:
                    if key in overall_metrics:
                        f.write(f"  {key}: {overall_metrics[key]:.4f}\n")

                f.write("\n")


        f.write(f"\n{'=' * 60}\n")
        f.write("Cross-seed statistics\n")
        f.write(f"{'=' * 60}\n\n")


        metric_types = sorted(set(r['metric_name'] for r in all_test_results))

        for metric_type in metric_types:
            type_results = [r for r in all_test_results if r['metric_name'] == metric_type]
            if len(type_results) > 1:
                f.write(f"\n{metric_type} Cross-seed performance:\n")


                for key in ['image_to_text_R@1', 'text_to_image_R@1']:
                    values = []
                    for result in type_results:
                        metrics = result['metrics']
                        overall_metrics = {k: v for k, v in metrics.items() if k != "per_sample_metrics"}
                        if key in overall_metrics:
                            values.append(overall_metrics[key])

                    if values:
                        f.write(f"  {key}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                                f"range=[{min(values):.4f}, {max(values):.4f}]\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of report\n")
        f.write("=" * 80 + "\n")

    print(f"Detailed test report saved to: {report_path}")


if __name__ == "__main__":
    main()