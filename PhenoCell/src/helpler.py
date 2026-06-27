"""
Utility function and class
[1] https://github.com/openai/CLIP/issues/111
"""
import glob
import os
import sys
from functools import partial
from multiprocessing import Pool
import safetensors.torch
import numpy as np
import torch
import torch.distributed as dist
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

from configs.model_config import ModelConfig
from src.clip.model import (
    BERT_CLIP,
    CellCLIP,
    CellCLIPOri,
    CellCLIP_MAE,
    CLIP_ChannelViT,
    CLIP_ResNet,
    CellCLIPOri,
    Cloome,
    Cloome_MPNN,
    Cloome_old,
    Cloome_phenom1,
    Molphenix,
)


def compute_model_stats(model, input_size=(3, 224, 224)):
    """
    Compute number of parameters and FLOPs for a given model.

    Args:
    ----
        model (torch.nn.Module): Pretrained model (e.g., DINOv2 ViT).
        input_size (tuple): Input tensor size, default (3, 224, 224).

    Return:
    ------
        params (float): Total parameters (in millions).
        flops (float): Total FLOPs (in billions).
    """
    model.eval()  # set to eval mode
    dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)

    # Compute FLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total() / 1e9  # convert to GFLOPs (GigaFLOPs)

    # Compute Parameters
    params_table = parameter_count_table(model)
    params = sum(p.numel() for p in model.parameters()) / 1e6  # convert to millions

    print(params_table)  # optional: shows nice table

    return params, total_flops


def parallelize(func, iterable, n_workers, **kwargs):
    """Helper function for parallelization"""
    f = partial(func, **kwargs)
    if n_workers > 1:
        with Pool(n_workers) as p:
            results = p.map(f, iterable)
    else:
        results = list(map(f, iterable))
    return results


def compute_grad_norm(accelerator, model):
    """Compute gradient norm. To be run under the accelerator main process."""
    model = accelerator.unwrap_model(model)
    grads = [
        param.grad.detach().cpu().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.cat(grads).norm()


def compute_param_norm(accelerator, model):
    """Compute the parameter norm. To be run under the accelerator main process."""
    model = accelerator.unwrap_model(model)
    params = [
        param.data.detach().cpu().flatten()
        for param in model.parameters()
        if param.data is not None
    ]
    return torch.cat(params).norm()


def get_max_steps(folder_path):
    """Get maximum number of training steps for results in a folder."""

    path_pattern = os.path.join(folder_path, "ckpt_steps_*.pt")
    files = glob.glob(path_pattern)

    if not files:
        return None

    max_steps = max(
        files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
    )
    return int(os.path.basename(max_steps).split("_")[-1].split(".")[0])


def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


class AllGatherFunction(torch.autograd.Function):
    """
    Custom autograd function for distributed training that performs an all-gather
    on input tensors across all nodes during the forward pass and sums
    then scatters gradients during the backward pass.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)


def all_gather(tensor):
    """Wrapper function for all-gather."""
    return AllGatherFunction.apply(tensor)


def get_metrics_recall(image_features, text_features, batch_size=512):
    """Evaluate retrieval with batching to avoid memory issues."""
    metrics = {}

    n_samples = len(image_features)
    device = image_features.device


    image_features = image_features.detach()
    text_features = text_features.detach()


    for direction_name, query_features, key_features in [
        ("image_to_text", image_features, text_features),
        ("text_to_image", text_features, image_features)
    ]:
        all_ranks = []


        for i in range(0, n_samples, batch_size):

            query_batch = query_features[i:i + batch_size]
            batch_size_actual = len(query_batch)


            for k in range(batch_size_actual):

                query_vec = query_batch[k:k + 1]  # [1, feature_dim]


                similarity_scores = []
                for j in range(0, n_samples, batch_size):
                    key_batch = key_features[j:j + batch_size]

                    sim = query_vec @ key_batch.t()  # [1, batch_size_key]
                    similarity_scores.append(sim)


                similarity_scores = torch.cat(similarity_scores, dim=1)  # [1, n_samples]


                sorted_indices = torch.argsort(similarity_scores[0], descending=True)


                correct_index = i + k
                rank = torch.where(sorted_indices == correct_index)[0]

                if len(rank) > 0:
                    all_ranks.append(rank[0].item())
                else:
                    all_ranks.append(n_samples)


        all_ranks = np.array(all_ranks)


        metrics[f"{direction_name}_mean_rank"] = all_ranks.mean() + 1
        metrics[f"{direction_name}_median_rank"] = np.floor(np.median(all_ranks)) + 1
        for k in [1, 5, 10]:
            metrics[f"{direction_name}_R@{k}"] = np.mean(all_ranks < k)

    return metrics




def get_metrics(image_features, text_features, logit_scale=None):
    """EvaluationRetrievalMetric，Precision@K"""


    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)


    if logit_scale is not None:
        similarity_matrix = logit_scale * image_features_norm @ text_features_norm.t()
    else:
        similarity_matrix = image_features_norm @ text_features_norm.t()

    logits_per_image = similarity_matrix
    logits_per_text = similarity_matrix.t()

    metrics = {}


    metrics.update(get_metrics_recall(image_features, text_features))


    metrics.update(compute_top_k_percent_recall(similarity_matrix))


    metrics.update(compute_ranking_quality_metrics(similarity_matrix))


    metrics.update(compute_zero_shot_classification_metrics(similarity_matrix))

    return metrics


def compute_top_k_percent_recall(similarity_matrix):
    """Top k% recall accuracy (k=1%, 5%, 10%)"""
    metrics = {}


    n_samples = similarity_matrix.shape[0]


    img_to_text_topk_percent = compute_top_k_percent_for_direction(similarity_matrix, "image_to_text")
    metrics.update(img_to_text_topk_percent)


    text_to_img_topk_percent = compute_top_k_percent_for_direction(similarity_matrix.T, "text_to_image")
    metrics.update(text_to_img_topk_percent)


    # metrics.update(compute_average_top_k_percent_recall(metrics))

    return metrics


def compute_top_k_percent_for_direction(similarity_matrix, direction_name):
    """Top k% recall accuracy"""
    n_samples = similarity_matrix.shape[0]
    similarity_np = similarity_matrix.detach().cpu().numpy()


    topk_percent_metrics = {}


    k_percent_values = [1, 5, 10]  # 1%, 5%, 10%


    for k_percent in k_percent_values:

        k_candidates = max(1, int(np.ceil(n_samples * k_percent / 100)))


        correct_count = 0

        for i in range(n_samples):

            scores = similarity_np[i]


            top_k_candidates = np.argsort(-scores)[:k_candidates]


            if i in top_k_candidates:
                correct_count += 1


        accuracy = correct_count / n_samples
        topk_percent_metrics[f"{direction_name}_Top{k_percent}%_Recall"] = accuracy

    return topk_percent_metrics


def compute_basic_retrieval_metrics_original(logits_per_image, logits_per_text):
    """Basic retrieval metrics"""
    metrics = {}

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(logits_per_text)).view(-1, 1)

    for name, logit in logits.items():
        logit = logit.detach().cpu()
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def compute_ranking_quality_metrics(similarity_matrix):
    """Ranking quality metrics：mAP, NDCG"""
    metrics = {}

    similarity_np = similarity_matrix.detach().cpu().numpy()
    n_samples = similarity_np.shape[0]


    img_to_text_map, img_to_text_ndcg = compute_map_and_ndcg(similarity_np, "image_to_text")
    metrics.update(img_to_text_map)
    metrics.update(img_to_text_ndcg)


    text_to_img_map, text_to_img_ndcg = compute_map_and_ndcg(similarity_np.T, "text_to_image")
    metrics.update(text_to_img_map)
    metrics.update(text_to_img_ndcg)

    return metrics


def compute_map_and_ndcg(similarity_matrix, direction_name):
    """HourmAPNDCG@10"""
    n_samples = similarity_matrix.shape[0]


    aps = []
    ndcgs = []

    for i in range(n_samples):

        scores = similarity_matrix[i]


        relevance = np.zeros(n_samples)
        relevance[i] = 1


        ap = compute_average_precision(scores, relevance)
        aps.append(ap)


        ndcg = compute_ndcg_at_k(scores, relevance, k=10)
        ndcgs.append(ndcg)


    metrics_map = {f"{direction_name}_mAP": np.mean(aps)}
    metrics_ndcg = {f"{direction_name}_NDCG@10": np.mean(ndcgs)}

    return metrics_map, metrics_ndcg


def compute_average_precision(scores, relevance):
    """Precision（AP）"""

    sorted_indices = np.argsort(-scores)
    sorted_relevance = relevance[sorted_indices]


    precision_at_k = []
    relevant_count = 0

    for k, rel in enumerate(sorted_relevance, 1):
        if rel > 0:
            relevant_count += 1
            precision_at_k.append(relevant_count / k)

    if precision_at_k:
        return np.mean(precision_at_k)
    else:
        return 0.0


def compute_ndcg_at_k(scores, relevance, k=10):
    """NDCG@K"""

    sorted_indices = np.argsort(-scores)[:k]
    sorted_relevance = relevance[sorted_indices]


    dcg = 0
    for i, rel in enumerate(sorted_relevance, 1):
        dcg += rel / np.log2(i + 1)


    ideal_relevance = np.sort(relevance)[::-1][:k]
    idcg = 0
    for i, rel in enumerate(ideal_relevance, 1):
        idcg += rel / np.log2(i + 1)


    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_zero_shot_classification_metrics(similarity_matrix):
    """Zero-shot classification metrics"""
    metrics = {}

    similarity_np = similarity_matrix.detach().cpu().numpy()
    n_samples = similarity_np.shape[0]


    predictions = np.argmax(similarity_np, axis=1)


    true_labels = np.arange(n_samples)


    top1_acc = np.mean(predictions == true_labels)
    metrics["zero_shot_top1_accuracy"] = top1_acc


    top5_correct = 0
    for i in range(n_samples):
        top5_indices = np.argsort(-similarity_np[i])[:5]
        if i in top5_indices:
            top5_correct += 1
    top5_acc = top5_correct / n_samples
    metrics["zero_shot_top5_accuracy"] = top5_acc


    top10_correct = 0
    for i in range(n_samples):
        top10_indices = np.argsort(-similarity_np[i])[:10]
        if i in top10_indices:
            top10_correct += 1
    top10_acc = top10_correct / n_samples
    metrics["zero_shot_top10_accuracy"] = top10_acc


    if n_samples > 10:

        class_accuracies = []
        for label in true_labels:
            mask = true_labels == label
            if mask.sum() > 0:
                class_acc = np.mean(predictions[mask] == label)
                class_accuracies.append(class_acc)

        metrics["zero_shot_mean_class_accuracy"] = np.mean(class_accuracies)
        metrics["zero_shot_std_class_accuracy"] = np.std(class_accuracies)

    return metrics



def get_metrics_with_enrichment(image_features, text_features, matches_matrix=None,
                                logit_scale=None, top_percent=0.01):
    """
    MinuteEvaluationMetric
    """
    metrics = get_metrics(image_features, text_features, logit_scale)


    if matches_matrix is not None:
        similarity_matrix = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity_matrix = similarity_matrix @ (text_features / text_features.norm(dim=-1, keepdim=True)).t()
        if logit_scale is not None:
            similarity_matrix = logit_scale * similarity_matrix

        enrichment_metrics = calculate_folds_of_enrichment(
            similarity_matrix.detach().cpu().numpy(),
            matches_matrix,
            top_percent=top_percent
        )
        metrics.update(enrichment_metrics)

    return metrics


def calculate_folds_of_enrichment(similarity_matrix, matches_matrix, top_percent=0.01):
    """
    （Folds of Enrichment, FoE）
    """
    n_samples = similarity_matrix.shape[0]
    top_k = max(1, int(n_samples * top_percent))

    all_odds_ratios = []
    all_p_values = []


    for i in range(n_samples):

        scores = similarity_matrix[i].copy()
        matches = matches_matrix[i].copy()


        scores[i] = -np.inf
        matches[i] = False


        if np.sum(matches) == 0:
            continue


        sorted_indices = np.argsort(-scores)


        top_indices = sorted_indices[:top_k]
        bottom_indices = sorted_indices[top_k:]


        a = np.sum(matches[top_indices])
        b = top_k - a
        c = np.sum(matches[bottom_indices])
        d = len(bottom_indices) - c


        if a == 0 or b == 0 or c == 0 or d == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5


        odds_ratio = (a * d) / (b * c)


        try:
            odds_ratio_test, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        except:
            odds_ratio_test, p_value = odds_ratio, 1.0

        all_odds_ratios.append(odds_ratio)
        all_p_values.append(p_value)


    if len(all_odds_ratios) == 0:
        return {
            'FoE_mean': 0.0,
            'FoE_median': 0.0,
            'FoE_std': 0.0,
            'FoE_geometric_mean': 0.0,
            'FoE_significant_fraction': 0.0,
        }


    log_odds = np.log(all_odds_ratios)
    geometric_mean_odds = np.exp(np.mean(log_odds))


    significant_fraction = np.mean([p < 0.05 for p in all_p_values])

    return {
        'FoE_mean': float(np.mean(all_odds_ratios)),
        'FoE_median': float(np.median(all_odds_ratios)),
        'FoE_std': float(np.std(all_odds_ratios)),
        'FoE_geometric_mean': float(geometric_mean_odds),
        'FoE_significant_fraction': float(significant_fraction),
    }



def load(model_path, device, model_type, input_dim=768, loss_type="clip"):
    """Load pretrained model from checkpoint."""
    MODEL_CONFIGS = {
        "old_cloome": (Cloome_old, ModelConfig.old_cloome_config),
        "cloome": (Cloome, ModelConfig.cloome_config),
        "cloome_mpnn": (Cloome_MPNN, ModelConfig.cloome_mpnn_config),
        "bert_clip": (BERT_CLIP, ModelConfig.bert_clip_config),
        "clip_resnet": (CLIP_ResNet, ModelConfig.clip_resnet_config),
        "clip_channelvit": (CLIP_ChannelViT, ModelConfig.clip_channelvit_config),
        # "pubmed_clip": (Pubmed_CLIP, ModelConfig.pubmed_clip_config),
        "cell_clip": (CellCLIP, ModelConfig.cell_clip_config),        
        "cell_clip_mae": (CellCLIP_MAE, ModelConfig.cell_clip_mae_config),
        "cloome_phenom1": (Cloome_phenom1, ModelConfig.cloome_phenom1_config),
        "molphenix": (Molphenix, ModelConfig.molphenix_config),
    }

    # Special handling for new_cell_clip due to dynamic config
    if model_type == "cell_clip":

        model_config = ModelConfig.cell_clip_config.copy()
        model_config["vision_width"] = input_dim
        model_config["use_bias"] = True if loss_type in ["sigclip", "s2l"] else False

        model = CellCLIPOri(**model_config)

    elif model_type == "mil_cell_clip":

        model_config = ModelConfig.mil_cell_clip_config.copy()
        model_config["vision_width"] = input_dim
        model_config["use_bias"] = True if loss_type in ["sigclip", "s2l"] else False

        model = MilCellClip(**model_config)

    elif model_type == "pubmed_clip_phenom1":

        model_config = ModelConfig.pubmed_clip_phenom1_config.copy()
        model_config["vision_width"] = input_dim
        model = Pubmed_CLIP_phenom1(**model_config)
    else:
        try:
            ModelClass, config = MODEL_CONFIGS[model_type]
            model = ModelClass(**config)
        except KeyError:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types are: {list(MODEL_CONFIGS.keys()) + ['new_cell_clip']}"
            )

    # Load checkpoint
    file_ext = model_path.lower().split('.')[-1]
    try:
        if file_ext == 'safetensors':

            state_dict = safetensors.torch.load_file(model_path, device=device)
        elif file_ext in ['pt', 'pth']:

            checkpoint = torch.load(model_path, map_location=device)
            

            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {str(e)}")

    # Handle state dict format
    if model_type == "old_cloome":
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
    elif file_ext == 'safetensors':
        state_dict=state_dict
    else:
        state_dict = checkpoint["model"]

    # Convert to float32 if on CPU
    if str(device) == "cpu":
        model.float()

    # Load state dict and move to device
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict into model: {str(e)}")

    model.to(device)
    model.eval()

    print(f"Successfully loaded {model_type} model from {model_path}")

    return model
