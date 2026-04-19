"""
Run test evaluation with the best checkpoint.

Loads best_checkpoint_model.safetensors and evaluates on the test set.
Output: test_result.json (same format as dev eval + per-sample predictions).
"""

import os
import json
import argparse
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from src.config import (
    LLM_NAME, VIT_NAME, ASPECT_LABELS, NUM_ASPECTS,
    BATCH_SIZE, MAX_TEXT_LEN, DATA_DIR, OUTPUT_DIR,
)
from src.aspect_model import MultimodalACSAModel
from src.data import MultimodalSentimentDataset, collate_fn


SENTIMENT_LABELS = ["Irrelative", "Negative", "Neutral", "Positive"]


def parse_args():
    parser = argparse.ArgumentParser(description="Test evaluation with best checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file. Defaults to <output_dir>/best_checkpoint.pt")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    return parser.parse_args()


def load_model_weights(model: MultimodalACSAModel, ckpt_path: str, device: str = "cuda"):
    """Load model weights from safetensors."""
    from safetensors import safe_open

    safetensor_path = ckpt_path.replace(".pt", "_model.safetensors")
    if not os.path.exists(safetensor_path):
        raise FileNotFoundError(f"Model safetensors not found: {safetensor_path}")

    loaded_state = {}
    with safe_open(safetensor_path, framework="pt", device=device) as f:
        for key in f.keys():
            loaded_state[key] = f.get_tensor(key)

    def _extract_sub(state, prefix):
        return {k.replace(prefix + ".", ""): v for k, v in state.items() if k.startswith(prefix + ".")}

    lora_state = {k.replace("lora.", ""): v for k, v in loaded_state.items() if k.startswith("lora.")}
    if lora_state:
        model.llm.load_state_dict(lora_state, strict=False)

    for module_name, module in [
        ("mlp_projector", model.mlp_projector),
        ("roi_projector", model.roi_projector),
        ("head", model.head),
        ("aspect_queries", model.aspect_queries),
        ("text_retriever", model.text_retriever),
        ("img_retriever", model.img_retriever),
        ("patch_retriever", model.patch_retriever),
        ("roi_retriever", model.roi_retriever),
        ("gated_fusion", model.gated_fusion),
        ("img_sum_projector", model.img_sum_projector),
        ("presence_head", model.presence_head),
    ]:
        state = _extract_sub(loaded_state, module_name)
        if state:
            module.load_state_dict(state, strict=False)


def test_epoch(
    model: MultimodalACSAModel,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
) -> dict:
    """Evaluate on test set and collect per-sample predictions."""
    model.eval()
    aspect_preds = {i: [] for i in range(NUM_ASPECTS)}
    aspect_labels_acc = {i: [] for i in range(NUM_ASPECTS)}
    total_loss = 0.0
    num_batches = 0

    all_sample_preds = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Test Evaluation")
        for batch in pbar:
            comments = batch["comments"]
            pixel_values = batch["pixel_values"].to(device)
            aspect_labels = batch["aspect_labels"]
            image_names_batch = batch.get("image_names", [[] for _ in comments])

            encodings = tokenizer(
                comments,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_LEN,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            roi_data = batch["roi_data"]

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    roi_data=roi_data,
                    aspect_labels=aspect_labels,
                    image_mask=batch.get("image_mask"),
                    raw_images=batch.get("raw_images"),
                )

            total_loss += outputs["ce_loss"].item()

            logits = outputs["logits"]  # [B*6, 4]
            targets = outputs["targets"]  # [B*6]

            B = input_ids.size(0)
            for asp_idx in range(NUM_ASPECTS):
                start = asp_idx * B
                end = start + B
                preds = logits[start:end].argmax(dim=-1).cpu().tolist()
                labels = targets[start:end].cpu().tolist()
                aspect_preds[asp_idx].extend(preds)
                aspect_labels_acc[asp_idx].extend(labels)

            # Per-sample predictions
            logits_per_sample = logits.view(B, NUM_ASPECTS, -1)  # [B, 6, 4]
            probs_per_sample = torch.softmax(logits_per_sample, dim=-1).cpu().tolist()

            for b in range(B):
                sample_preds = {}
                for asp_idx, aspect_name in enumerate(ASPECT_LABELS):
                    pred_id = int(logits_per_sample[b][asp_idx].argmax().item())
                    prob = probs_per_sample[b][asp_idx]
                    sample_preds[aspect_name] = {
                        "prediction": SENTIMENT_LABELS[pred_id],
                        "prediction_id": pred_id,
                        "confidence": float(prob[pred_id]),
                        "probabilities": {
                            SENTIMENT_LABELS[c]: float(prob[c]) for c in range(4)
                        },
                    }

                all_sample_preds.append({
                    "comment": comments[b],
                    "image_names": image_names_batch[b],
                    "predictions": sample_preds,
                })

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    all_preds = []
    all_labels = []
    for asp_idx in range(NUM_ASPECTS):
        all_preds.extend(aspect_preds[asp_idx])
        all_labels.extend(aspect_labels_acc[asp_idx])

    overall_f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    overall_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_accuracy = accuracy_score(all_labels, all_preds)

    per_aspect = {}
    for asp_idx, aspect_name in enumerate(ASPECT_LABELS):
        preds = aspect_preds[asp_idx]
        labels = aspect_labels_acc[asp_idx]
        if len(set(labels)) > 0:
            per_aspect[aspect_name] = {
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
                "precision": precision_score(labels, preds, average="macro", zero_division=0),
                "recall": recall_score(labels, preds, average="macro", zero_division=0),
                "accuracy": accuracy_score(labels, preds),
                "f1_per_class": f1_score(labels, preds, average=None, zero_division=0).tolist(),
            }
        else:
            per_aspect[aspect_name] = {
                "f1_macro": 0.0, "f1_weighted": 0.0,
                "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
                "f1_per_class": [0.0, 0.0, 0.0, 0.0],
            }

    return {
        "loss": avg_loss,
        "overall_f1_macro": overall_f1_macro,
        "overall_f1_weighted": overall_f1_weighted,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_accuracy": overall_accuracy,
        "per_aspect": per_aspect,
        "per_sample_predictions": all_sample_preds,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = output_dir / "best_checkpoint.pt"

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"Looking for: {ckpt_path}")
        return

    safetensor_path = str(ckpt_path).replace(".pt", "_model.safetensors")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Model weights: {safetensor_path}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading model...")
    model = MultimodalACSAModel(use_lora=args.use_lora)
    model.to(device)
    load_model_weights(model, str(ckpt_path), device=str(device))
    print("Weights loaded successfully.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load test dataset
    test_dataset = MultimodalSentimentDataset(
        split="test",
        data_dir=args.data_dir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    print(f"Test set: {len(test_dataset)} samples")

    # Run evaluation
    print()
    results = test_epoch(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
    )

    # Print results
    print()
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test F1 (macro): {results['overall_f1_macro']:.4f}  "
          f"Precision: {results['overall_precision']:.4f}  "
          f"Recall: {results['overall_recall']:.4f}  "
          f"Accuracy: {results['overall_accuracy']:.4f}")
    print("Per-aspect metrics (F1 / Precision / Recall / Acc):")
    for aspect_name, metrics in results["per_aspect"].items():
        print(f"  {aspect_name}: F1={metrics['f1_macro']:.4f}  "
              f"P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}")

    # Save to test_result.json (without per_sample_predictions)
    result_to_save = {k: v for k, v in results.items() if k != "per_sample_predictions"}
    output_path = output_dir / "test_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_to_save, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
