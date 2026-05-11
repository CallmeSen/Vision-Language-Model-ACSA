# Vision-Language Model ACSA: Architecture for Vietnamese Multimodal Aspect-Category Sentiment Analysis

# Overview

- Propose a **Vision-Language Model ACSA** architecture that processes text and images jointly with explicit cross-modal retrieval and gated fusion for each aspect independently.

# Table of Contents

- [Dataset](#dataset)
- [Architecture](#architecture)
- [Running The Code](#running-the-code)
- [Experiment Results](#experiment-results)

# Dataset
To understand more about the dataset, please read this paper: [New Benchmark Dataset and Fine-Grained Cross-Modal Fusion Framework for Vietnamese Multimodal Aspect-Category Sentiment Analysis](https://arxiv.org/abs/2405.00543)

This dataset is used only for research purposes. Download the ViMACSA dataset on reasonable request: https://drive.google.com/file/d/1OjWwzdbhvhYc864Tpt6Xw9anBLfgNwmt/view?usp=sharing

> **Note**: This repository builds upon the original [ViMACSA dataset](https://github.com/hoangquy18/Multimodal-Aspect-Category-Sentiment-Analysis) by Hoang Quy et al. (2025).
**6 Aspects**: Facilities, Public_area, Location, Food, Room, Service

**4 Sentiment Classes**: Irrelative (0), Negative (1), Neutral (2), Positive (3)

## Dataset Statistics

| Split     | Samples | Images | Annotations |
|-----------|---------|--------|-------------|
| Train     | ~3,600  | ~9,000 | ~11,000     |
| Dev       | ~500    | ~1,500 | ~1,800      |
| Test      | ~500    | ~1,500 | ~1,800      |
| **Total** | **4,876** | **~12,000** | **~14,618** |

# Architecture

We propose the **VLM-ACSA (Vision Language Model Aspect-Category Sentiment Analysis)** architecture with two core design principles:

1. **Encode once**: comment, all images, and all ROIs are encoded only once per forward pass
2. **Aspect loop**: shared retrieval/fusion/decoder modules process each aspect independently

## Model Backbones

- **Vision backbone**: SigLIP2-Large (`google/siglip2-so400m-patch16-256`), frozen
- **LLM backbone**: Qwen/Qwen3-4B-Instruct-2507, fine-tuned with LoRA (rank=64)

## Pipeline

```
Input: comment + images [B, M, 3, 256, 256] + roi_data per image
  |
  v
ENCODING (all frozen)
  comment ──► llm.embed_tokens() ─────► H_txt [B, L_txt, 2560]
  images ───► SigLIPEncoder ──► patch_feats [B, M, P, 1152]
               │              img_summaries [B, M, 1152]
               └─► encode_roi(pixel_values, roi_data)
                                       └─► roi_seq [B, M, K, 1152]
  |
  v
PROJECTION (trainable)
  patch_feats ──► MLPProjector ──► H_patch [B, M, P', 2560]
  roi_seq ───────► RoIProjector ──► H_roi [B, M, K', 2560]
  img_summaries ─► Linear(1152→2560) ──► H_img_sum [B, M, 2560]
  |
  v
ASPECT LOOP (shared weights, 6 iterations)
  For each aspect a:
    aspect_queries[a] ────────────────────────────► ASP_a [B, 2560]
    H_txt ──► TextRetriever (8-head cross-attn) ─► TXT_EVI_a [B, 2560]
    H_img_sum ──► ImageRetriever (4-head cross-attn) ─► IMG_EVI_a [B, 2560]
    H_roi ──► RoiRetriever (4-head cross-attn) ─► ROI_EVI_a [B, 2560]
    [TXT_EVI | IMG_EVI | ROI_EVI] ──► GatedFusion ──► FUSE_a [B, 2560]
    [ASP | TXT | IMG | ROI | FUSE] ──► Qwen3-4B + LoRA ──► logits_a [B, 4]
  |
  v
CrossEntropyLoss on logits [B*6, 4] vs targets [B*6]
```

## Key Innovations vs Baseline

| Aspect | Baseline (MultimodalSentimentModel) | ViMACSA |
|--------|-------------------------------------|---------|
| Sequence | All tokens concat | Per-aspect 5-token sequence |
| Text retrieval | Not explicit | Cross-attention from aspect query |
| Image retrieval | Flat concat into LLM | Cross-attention + relevance weights |
| ROI retrieval | Flat concat into LLM | Cross-attention + relevance weights |
| Fusion | None (LLM handles) | Learnable gated fusion |
| Aspect handling | Single LLM forward | Loop 6× with shared decoder |

## Trainable Parameters

| Component | Parameters |
|-----------|-----------|
| LoRA (Qwen3-4B) | ~23.6M |
| Projection layers | ~193.1M |
| Retrievers + Fusion | ~15K |
| **Total trainable** | **~216.7M** |

# Running The Code

## Install Requirements

```
pip install -r requirements.txt
```

## Training VLM-ACSA

```
torchrun --standalone --nproc_per_node=1 training.py
```

- Optimizer: AdamW with 2 groups (LoRA lr=2e-5, other lr=3e-5)
- Scheduler: warmup + cosine
- Early stopping: patience=4 on dev macro-F1

## Evaluation

```
torchrun --standalone --nproc_per_node=1 test_eval.py
```

## Inference

```python
from inference import predict_all_aspects

results = predict_all_aspects(
    model=model,
    tokenizer=tokenizer,
    comment="Khách sạn này rất đẹp, nhân viên nhiệt tình.",
    image_paths=["img1.jpg", "img2.jpg"],
    roi_data=None  # optional per-image ROI dicts
)
# Returns: { aspect_name: {prediction, prediction_id, confidence, probabilities} }
```

# Experiment Results

## Training Progress

| Epoch | Train Loss | Dev Loss | Dev F1 (macro) | Precision | Recall | Accuracy |
|-------|-----------|----------|----------------|-----------|--------|----------|
| 1     | 1.1046    | 0.8810   | 0.4021         | 0.4778    | 0.4004 | 0.6602   |
| 2     | 0.8841    | 0.7655   | 0.4952         | 0.6425    | 0.4705 | 0.6933   |
| 3     | 0.7052    | 0.6202   | 0.6120         | 0.6076    | 0.6253 | 0.7618   |
| 4     | 0.5620    | 0.4998   | 0.5670         | 0.7168    | 0.5405 | 0.8112   |
| 5     | 0.4677    | 0.4853   | 0.6320         | 0.6899    | 0.6050 | 0.8265   |
| 6     | 0.4012    | 0.4715   | 0.6489         | 0.6812    | 0.6332 | 0.8322   |
| **7** | **0.3211** | **0.5000** | **0.6507**     | **0.6948** | **0.6263** | **0.8340** |
| 11    | 0.1052    | 0.7702   | 0.6491         | 0.6913    | 0.6245 | 0.8277   |

*Early stopping triggered at epoch 11 (best model saved from epoch 7).*

## Test Set Performance

| Metric | Value |
|--------|-------|
| **Macro-F1** | **0.6538** |
| Precision | 0.6893 |
| Recall | 0.6316 |
| Accuracy | 0.8262 |

## Per-Aspect F1 (Test Set)

| Aspect | F1 | Precision | Recall | Accuracy |
|--------|-----|-----------|--------|----------|
| Facilities | 0.5457 | 0.5644 | 0.5309 | 0.8040 |
| Public_area | 0.5456 | 0.5880 | 0.5421 | 0.7700 |
| Location | 0.4426 | 0.4729 | 0.4419 | 0.8750 |
| Food | 0.5911 | 0.6327 | 0.5634 | 0.8590 |
| Room | 0.7114 | 0.7205 | 0.7041 | 0.7830 |
| Service | 0.5933 | 0.5998 | 0.5919 | 0.8660 |
