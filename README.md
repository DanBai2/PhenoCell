# PhenoCell for cellular morphology-based drug response prediction

This repository provides the official implementation of **PhenoCell**, a morphology-based virtual-cell framework that uses multiplex Cell Painting imaging as a computable state language. 
PhenoCell combines a multi-channel morphology encoder with State-conditioned Phenotypic Response Alignment (SPRA), which conditions chemical-structure representations on the control-cell phenotype to learn state-conditioned perturbation-response representations.

---

## Directory Structure

```
├── src/                        # Core source code
│   ├── clip/                   # Model architectures, contrastive losses, tokenizer
│   ├── channelvit/             # Channel Vision Transformer backbone
│   ├── open_phenom/            # OpenPhenom MAE integration
│   ├── mpnn/                   # MPNN molecule encoder
│   ├── benchmark/              # CP-JUMP1 benchmark utilities
│   ├── tools/                  # Data download and statistics utilities
│   ├── transformations/        # Image transformation pipelines
│   └── datasets.py / helpler.py / scheduler.py / constants.py
│
├── scripts/                    # Executable scripts
│   ├── train/                  # Training entry points
│   ├── eval/                   # Evaluation and retrieval
│   └── preprocessing/          # Data preprocessing pipelines
│
├── configs/                    # Model, data, and DDP configuration
├── vendor/                     # Vendored third-party dependencies
├── setup.py / requirements.txt
└── README.md
```

---

## Quick Start: Evaluate Pretrained Model on Your Data

The primary entry point for using PhenoCell with your own data is `scripts/eval/test.py`. It loads a pretrained model checkpoint and computes retrieval metrics on your test set.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

You need to prepare three parts of data before training or evaluation:

#### Part 1: Image Data (npz → Embeddings)

**Step 1.1 — Preprocess raw images into .npz format:**

We provide multiple scripts for different data formats. Choose the one that matches your input:

| Script | Use Case |
|--------|----------|
| `scripts/preprocessing/preprocess_images_ours.py` | Custom in-house image format |
| `scripts/preprocessing/preprocess_images.py` | Standard Cell Painting images |
| `scripts/preprocessing/preprocess_images_rxrx19a.py` | RxRx19a dataset format |
| `scripts/preprocessing/preprocess_images_jumpcp.py` | CP-JUMP1 dataset format |
| `scripts/preprocessing/preprocess_images_rxrx3.py` | RxRx3-core dataset format |

**Step 1.2 — Convert .npz to embeddings:**

```bash
python scripts/preprocessing/convert_npz_to_avg_emb.py \
    --model_card facebook/dinov2-giant \     # Feature extractor (facebook/dinov2-giant, facebook/dino-vitb8, etc.)
    --dataset your_dataset_name \
    --input_dir path/to/your/preprocessed_npz \
    --dataset_dir path/to/your/output \
    --output_file embeddings.h5
```

#### Part 2: Drug/Molecule Data (CAS → SMILES → Morgan Fingerprints)

**Step 2.1 — Convert CAS to SMILES:**

Use `scripts/preprocessing/convert_cas_to_smiles.py` to query PubChem and convert CAS numbers to canonical SMILES:

```bash
python scripts/preprocessing/convert_cas_to_smiles.py
```

Edit `ANNOTATION_PATH` and `OUTPUT_DIR` in the script to point to your annotation file and desired output directory.

**Step 2.2 — Convert SMILES to Morgan fingerprints:**

```bash
python scripts/preprocessing/convert_smiles_to_morgan_hd5.py \
    --data_dir path/to/your/smiles_data \
    --hdf5_output path/to/your/output/fingerprints.hdf5 \
    --cell_type YOUR_CELL_TYPE
```

This generates a `.hdf5` file containing 1024-bit Morgan fingerprints keyed by sample ID.

#### Part 3: Metadata Splitting (ID / OOD)

Split your metadata into train/val/test sets. We support two evaluation modes:

- **Seen (ID)**: Treatments in val/test also appear in train (random sample-level split)
- **Unseen (OOD)**: Treatments in val/test are disjoint from train (treatment-level split)

Refer to `scripts/preprocessing/rxrx19a_random_datasets_split.py` as a template — it processes `metadata.csv` grouped by cell type and generates both seen and unseen split CSV files.

After running the split, your data directory should look like:

```
path/to/your/data/
├── embeddings/
│   └── dinov2-giant_ind_HRCE_1536.h5         # Image embeddings
├── fingerprints/
│   └── HRCE_morgan_chiral_fps_1024.hdf5      # Morgan fingerprints
├── splits/
│   ├── HRCE-seen-train.csv
│   ├── HRCE-seen-val.csv
│   ├── HRCE-seen-test.csv
│   ├── HRCE-unseen-train.csv
│   ├── HRCE-unseen-val.csv
│   └── HRCE-unseen-test.csv
└── metadata.csv
```

### 3. Run Evaluation

```bash
python scripts/eval/test.py \
    --training_results_dir path/to/your/trained_model_dir \
    --model_type pheno_cell \
    --input_dim 1536 \
    --dataset your_dataset \
    --dataset_mode unseen \
    --cell_type YOUR_CELL_TYPE \
    --img_dir path/to/your/data/embeddings.h5 \
    --molecule_path path/to/your/data/fingerprints.hdf5 \
    --mole_struc morgan \
    --loss_type clip \
    --batch_size 64 \
    --test_seeds 42 123 456
```


### 4. Output

`test.py` produces the following in `{training_results_dir}/test_results/`:

```
test_results/
├── test_config.json              # Test configuration
├── overall_metrics.json          # Aggregate retrieval metrics
├── per_sample_metrics.json       # Per-sample detailed metrics
├── per_sample_results.csv        # CSV summary for all samples
├── all_models_test_summary.csv   # Cross-model comparison
└── detailed_test_report.txt      # Human-readable report
```

**Metrics reported**: R@1, R@5, R@10, Top k% Recall, mAP, NDCG@10, zero-shot accuracy (image-to-text and text-to-image for each).

---

## Training from Scratch

### Preprocessing

Follow the same data preparation pipeline described in [Quick Start > Prepare Your Data](#2-prepare-your-data) above:

1. **Image**: Raw images → npz (`preprocess_images_*.py`) → embeddings (`convert_npz_to_avg_emb.py`)
2. **Drug/Molecule**: CAS → SMILES (`convert_cas_to_smiles.py`) → Morgan fingerprints (`convert_smiles_to_morgan_hd5.py`)
3. **Metadata**: Split into train/val/test in seen (ID) or unseen (OOD) mode (`rxrx19a_random_datasets_split.py`)

### Training

Once data is prepared, run `scripts/train/main.py`:

```bash
python scripts/train/main.py \
    --batch_size 256 \
    --return_control \
    --use_true_control \
    --model_type "pheno_cell" \
    --img_dir "path/to/your/data/embeddings/dinov2-giant_ind_HRCE_1536.h5" \
    --mole_struc "morgan" \
    --molecule_path "path/to/your/data/embeddings/HRCE_morgan_chiral_fps_1024.hdf5" \
    --epochs 50 \
    --loss_type "clip" \
    --dataset "rxrx19a" \
    --dataset_mode "seen" \
    --cell_type "HRCE"
```

### Multi-GPU Training

```bash
accelerate launch --config_file configs/ddp_config.yaml scripts/train/main.py [args...]
```

---

## Supported Model Types

| Model Key | Architecture | Description |
|-----------|-------------|-------------|
| `pheno_cell` | PhenoCell | Cross-modal attention between text and control features |
| `cell_clip` | CellCLIP | Control feature fusion without cross-attention |
| `mil_cell_clip` | MIL-CellCLIP | Multi-instance learning variant |
| `cell_clip_mae` | CellCLIP-MAE | MAE-based vision encoder |
| `bert_clip` | BERT-CLIP | Pure BERT text-based CLIP |
| `clip_channelvit` | CLIP-ChannelViT | Channel-aware ViT CLIP |
| `molphenix` | MolPhenix | Baseline molecule-phenotype model |
| `cloome` | CLOOME | Baseline architecture |
| `cloome_mpnn` | CLOOME-MPNN | CLOOME with MPNN molecule encoder |

## Supported Loss Functions

| Loss Key | Description |
|----------|-------------|
| `clip` | Standard InfoNCE contrastive loss |
| `cwcl` | Cell-aware weighted contrastive loss |
| `bi_cwcl` | Bidirectional CWCL |
| `cloob` | Hopfield network-based contrastive loss |
| `sigclip` | Sigmoid-based CLIP loss |
| `s2l` | S2L loss |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{phenocell2025,
  title={PhenoCell for cellular morphology-based drug response prediction},
  author={...},
  journal={...},
  year={2026}
}
```

## License

This project is released for academic research purposes. See LICENSE file for details.
