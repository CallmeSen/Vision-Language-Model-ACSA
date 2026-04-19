import torch
import torch.nn as nn
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

from src.config import VIT_NAME, VIT_HIDDEN, IMAGE_SIZE


ROI_CROP_SIZE = 256


class SigLIPEncoder(nn.Module):
    """SigLIP2-Large encoder — shared for patch and RoI encoding."""

    def __init__(self, model_name: str = VIT_NAME, image_size: int = IMAGE_SIZE):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.image_size = image_size

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.hidden_size = self._get_hidden_size()
        self.num_patches = self._compute_num_patches()

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _get_hidden_size(self) -> int:
        config = self.model.config
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            return config.vision_config.hidden_size
        return VIT_HIDDEN

    def _get_patch_size(self) -> int:
        config = self.model.config
        if hasattr(config, "patch_size"):
            return config.patch_size
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            return config.vision_config.patch_size
        return 16

    def _compute_num_patches(self) -> int:
        patch_size = self._get_patch_size()
        return (self.image_size // patch_size) ** 2

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, M, 3, H, W] — normalized via SigLIP processor
        Returns:
            patch_tokens: [B, M, P, D_v]
            img_summaries: [B, M, D_v]
        """
        B, M, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * M, C, H, W)

        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values_flat)
            last_hidden = outputs.last_hidden_state

        seq_len = last_hidden.shape[1]

        if seq_len == self.num_patches + 1:
            img_summary = last_hidden[:, 0, :]
            patches = last_hidden[:, 1:, :]
        elif seq_len == self.num_patches:
            img_summary = last_hidden.mean(dim=1)
            patches = last_hidden
        else:
            import logging
            logging.getLogger(__name__).warning(
                f"Unexpected seq_len={seq_len}, expected {self.num_patches} or {self.num_patches+1}. "
                "Falling back to mean-pool."
            )
            img_summary = last_hidden.mean(dim=1)
            patches = last_hidden

        patch_tokens = patches.reshape(B, M, self.num_patches, self.hidden_size)
        img_summaries = img_summary.reshape(B, M, self.hidden_size)

        return patch_tokens, img_summaries

    # ── RoI encoding (shared vision backbone) ────────────────────────────────

    def _encode_full_image_for_roi(self, full_img_tensor: torch.Tensor) -> torch.Tensor:
        """Encode full image for RoI branch — returns pooled summary [D_v]."""
        if full_img_tensor.dim() == 3:
            full_img_tensor = full_img_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model.vision_model(full_img_tensor)
            last_hidden = outputs.last_hidden_state
            if last_hidden.size(1) > 1:
                summary = last_hidden[:, 0, :]
            else:
                summary = last_hidden.mean(dim=1)
        return summary.squeeze(0)

    def _process_roi_crops(self, raw_img: Image.Image, boxes: List[List[float]]) -> torch.Tensor:
        """Crop ROI regions from raw PIL image, process through SigLIP processor.
        Returns [K, 3, H, W] tensor of SigLIP-normalized ROI crops, or empty tensor."""
        if not boxes:
            return torch.zeros(0, 3, ROI_CROP_SIZE, ROI_CROP_SIZE, dtype=torch.float32)

        crops = []
        W_img, H_img = raw_img.size
        for box in boxes:
            x1, y1, x2, y2 = [float(c) for c in box]
            px1 = max(0, int(round(x1 * W_img)))
            py1 = max(0, int(round(y1 * H_img)))
            px2 = min(W_img, int(round(x2 * W_img)))
            py2 = min(H_img, int(round(y2 * H_img)))
            if px2 <= px1 or py2 <= py1:
                continue
            crop = raw_img.crop((px1, py1, px2, py2)).resize((ROI_CROP_SIZE, ROI_CROP_SIZE), Image.LANCZOS)
            inputs = self.processor(images=crop, return_tensors="pt")
            crops.append(inputs["pixel_values"][0])

        if not crops:
            return torch.zeros(0, 3, ROI_CROP_SIZE, ROI_CROP_SIZE, dtype=torch.float32)
        return torch.stack(crops)

    def _encode_roi_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """Encode RoI crops using the shared vision_model. Returns [K, D_v]."""
        if crops.size(0) == 0:
            return torch.zeros(0, self.hidden_size, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model.vision_model(crops)
            last_hidden = outputs.last_hidden_state
            if last_hidden.size(1) > 1:
                roi_feats = last_hidden[:, 0, :]
            else:
                roi_feats = last_hidden.squeeze(1)
        return roi_feats

    def encode_roi(
        self,
        pixel_values: torch.Tensor,
        roi_data: List[List[Dict[str, Any]]],
        raw_images: List[List[Optional[Image.Image]]] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, M, 3, H, W] — SigLIP-normalized tensors (padded slots = zeros)
            roi_data: List[List[dict]] — per sample, per image:
                [{"boxes": [[x1,y1,x2,y2], ...], "labels": [...]}, ...]
            raw_images: List[List[Optional[PIL.Image]]] — raw PIL images per sample per image slot.
                None entries for padded slots.

        Returns:
            roi_img_seq: [B, M, K_max, D_v]
                Token 0: pooled full-image summary [D_v]
                Token 1..K: per-RoI features [K, D_v]
                Padded with zeros to K_max.
        """
        B, M, C, H, W = pixel_values.shape
        D_v = self.hidden_size
        device = pixel_values.device

        # Detect padded slots: pixel_values == 0 means padded (SigLIP-processor zeros)
        padded_mask = (pixel_values.sum(dim=(2, 3, 4)) == 0)  # [B, M]

        # Determine K_max
        max_k = 1
        for sample_rois in roi_data:
            for img_roi in sample_rois:
                boxes = img_roi.get("boxes", [])
                max_k = max(max_k, len(boxes) + 1)

        K_max = max_k
        roi_img_seq = torch.zeros(B, M, K_max, D_v, device=device, dtype=torch.float32)

        for b in range(B):
            sample_rois = roi_data[b] if b < len(roi_data) else []
            sample_raw_imgs = raw_images[b] if raw_images and b < len(raw_images) else []
            for m in range(M):
                # Skip padded image slots
                if padded_mask[b, m]:
                    continue

                raw_img = sample_raw_imgs[m] if m < len(sample_raw_imgs) else None

                # Token 0: pooled full-image summary from pixel_values (SigLIP-normalized)
                summary_feat = self._encode_full_image_for_roi(
                    pixel_values[b, m].unsqueeze(0).to(device)
                )
                roi_img_seq[b, m, 0, :] = summary_feat

                img_roi = sample_rois[m] if m < len(sample_rois) else {"boxes": [], "labels": []}
                boxes = img_roi.get("boxes", [])
                if len(boxes) == 0:
                    continue

                # Crop ROIs from raw PIL and process through SigLIP processor
                if raw_img is not None:
                    crops = self._process_roi_crops(raw_img, boxes)
                else:
                    crops = torch.zeros(0, 3, ROI_CROP_SIZE, ROI_CROP_SIZE, dtype=torch.float32)

                if crops.size(0) == 0:
                    continue

                crops = crops.to(device)
                roi_feats = self._encode_roi_crops(crops)
                K_actual = min(roi_feats.size(0), K_max - 1)
                roi_img_seq[b, m, 1:1 + K_actual, :] = roi_feats[:K_actual]

        return roi_img_seq
