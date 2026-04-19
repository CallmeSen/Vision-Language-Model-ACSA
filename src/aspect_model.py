"""
Multimodal ACSA model — encode-once, aspect-loop architecture.

Key design:
  - Encode comment, images, ROIs only ONCE per forward pass.
  - Loop over 6 aspects with shared retrieval/fusion/decoder modules.
  - Per-aspect sequence: [ASP_a | TXT_EVI_a | IMG_EVI_a | ROI_EVI_a | FUSE_a]
  - Shared LLM decoder processes each 5-token sequence independently.
  - Final output: logits [B, 6, 4].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoModel, AutoTokenizer

from src.config import (
    LLM_NAME, VIT_NAME,
    LLM_HIDDEN, VIT_HIDDEN,
    LORA_R, LORA_ALPHA, LORA_TARGET_MODULES,
    NUM_ASPECTS,
    MAX_PATCH_TOKENS_PER_IMAGE,
    MAX_ROI_TOKENS_PER_IMAGE,
    PATCH_RETRIEVER_HEADS,
)
from src.vit_encoder import SigLIPEncoder
from src.projector_layer import MLPProjector, RoIProjector
from src.attention import (
    AspectQuery,
    TextRetriever,
    ImageRetriever,
    RoiRetriever,
    PatchRetriever,
    GatedFusion,
)
from src.aspect_sequence import build_aspect_sequence


class ClassificationHead(nn.Module):
    """Shared classification head — used by both legacy and ACSA models."""

    def __init__(self, d_h: int = LLM_HIDDEN, num_classes: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_h)
        self.fc1 = nn.Linear(d_h, d_h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_h, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm(h)
        h = self.fc1(h)
        h = self.act(h)
        return self.fc2(h)


class MultimodalACSAModel(nn.Module):
    """
    Encode-once ACSA model with aspect-guided retrieval and gated fusion.

    Processing pipeline per forward pass:
      1. Encode text  (frozen embed_tokens)
      2. Encode images via SigLIP → MLP → patch tokens + image summaries
      3. Encode ROIs via raw PIL crop → SigLIP processor → shared vision backbone
      4. For each aspect a (shared modules):
           a. Retrieve text evidence via TextRetriever
           b. Retrieve image evidence via ImageRetriever (image summaries)
           c. Retrieve patch evidence via PatchRetriever (fine-grained patches)
           d. Retrieve ROI evidence via RoiRetriever
           e. Fuse via GatedFusion
           f. Build 5-token sequence → shared LLM decoder → classification head
           g. Auxiliary presence head for aspect-absent detection
      5. Stack 6 aspect logits → CrossEntropy loss + auxiliary presence loss
    """

    def __init__(
        self,
        llm_name: str = LLM_NAME,
        vit_name: str = VIT_NAME,
        use_lora: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.llm_name = llm_name
        self.vit_name = vit_name
        self.use_lora = use_lora
        self.device = device
        self.dtype = dtype
        self.num_aspects = NUM_ASPECTS

        # ── Frozen LLM backbone ────────────────────────────────────────────────
        self.llm = AutoModel.from_pretrained(
            llm_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                task_type="FEATURE_EXTRACTION",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            # LoRA params are trainable by default; leave them as-is.
        else:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Frozen vision backbone ──────────────────────────────────────────────
        self.siglip = SigLIPEncoder(model_name=vit_name)
        for param in self.siglip.parameters():
            param.requires_grad = False
        self.siglip.eval()

        # ── Projectors ──────────────────────────────────────────────────────────
        self.mlp_projector = MLPProjector(d_v=VIT_HIDDEN, d_h=LLM_HIDDEN)
        self.roi_projector = RoIProjector(d_v=VIT_HIDDEN, d_h=LLM_HIDDEN)
        # Separate lightweight projector for pooled image summaries (no patch dim)
        self.img_sum_projector = nn.Linear(VIT_HIDDEN, LLM_HIDDEN)

        # ── Aspect query table ──────────────────────────────────────────────────
        self.aspect_queries = AspectQuery(d_h=LLM_HIDDEN, num_aspects=NUM_ASPECTS)

        # ── Retrieval modules ────────────────────────────────────────────────────
        self.text_retriever = TextRetriever(d_h=LLM_HIDDEN)
        self.img_retriever = ImageRetriever(d_h=LLM_HIDDEN)
        self.patch_retriever = PatchRetriever(d_h=LLM_HIDDEN, num_heads=PATCH_RETRIEVER_HEADS)
        self.roi_retriever = RoiRetriever(d_h=LLM_HIDDEN)

        # ── Fusion ───────────────────────────────────────────────────────────────
        self.gated_fusion = GatedFusion(d_h=LLM_HIDDEN)

        # ── Classification head ─────────────────────────────────────────────────
        self.head = ClassificationHead(d_h=LLM_HIDDEN)

        # ── Auxiliary presence head (Bug 9) ─────────────────────────────────────
        self.presence_head = nn.Sequential(
            nn.LayerNorm(LLM_HIDDEN),
            nn.Linear(LLM_HIDDEN, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    @property
    def model_device(self) -> torch.device:
        return self._dummy.device

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _get_embed_layer(self):
        if hasattr(self.llm, "get_input_embeddings"):
            return self.llm.get_input_embeddings()
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):
            return self.llm.model.embed_tokens
        raise AttributeError("Cannot find input embedding layer on the LLM.")

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embed_layer()(input_ids)

    def _get_img_mask(self, explicit_mask: torch.Tensor = None) -> torch.Tensor:
        """Return [B, M] mask: 1 for real images, 0 for padded.
        Uses explicit mask from dataset when available."""
        return explicit_mask  # [B, M]

    def _compress_patch_tokens(
        self, patch_tokens: torch.Tensor, max_tokens: int
    ) -> torch.Tensor:
        """
        patch_tokens: [B, M, P, D_h]
        Compress P patch tokens per image to max_tokens using adaptive pooling.
        """
        B, M, P, D = patch_tokens.shape
        if P <= max_tokens:
            return patch_tokens
        x = patch_tokens.reshape(B * M, P, D).transpose(1, 2)  # [BM, D, P]
        x = F.adaptive_avg_pool1d(x, max_tokens)
        x = x.transpose(1, 2).reshape(B, M, max_tokens, D)
        return x

    def _get_targets(
        self,
        B: int,
        aspect_labels: List[Dict[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        targets = torch.zeros(B, self.num_aspects, device=device, dtype=torch.long)
        for b in range(B):
            for asp_idx, label in aspect_labels[b].items():
                targets[b, asp_idx] = label
        return targets

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        roi_data: List[List[Dict[str, Any]]],
        aspect_labels: List[Dict[int, int]],
        image_mask: torch.Tensor = None,
        raw_images: List[List[Any]] = None,
    ) -> Dict[str, Any]:
        B = input_ids.size(0)
        device = input_ids.device

        # ── Step 1: Encode once ─────────────────────────────────────────────
        # Text: embed token IDs
        H_txt = self._embed_tokens(input_ids)  # [B, L_txt, D_h]

        # Images: SigLIP encode
        patch_feats, img_summaries = self.siglip(pixel_values)  # [B,M,P,D_v], [B,M,D_v]
        # Project patch tokens
        H_patch = self.mlp_projector(patch_feats)  # [B, M, P, D_h]
        H_patch = self._compress_patch_tokens(H_patch, MAX_PATCH_TOKENS_PER_IMAGE)

        # Project image summaries separately (no patch dim)
        H_img_sum = self.img_sum_projector(img_summaries)  # [B, M, D_h]

        # ROIs: encode then project (uses raw_images for SigLIP-normed crops)
        roi_seq = self.siglip.encode_roi(pixel_values, roi_data, raw_images)  # [B, M, K, D_v]
        H_roi = self.roi_projector(roi_seq)  # [B, M, K, D_h]

        # Flatten images and ROIs for retrieval
        H_patch_flat = H_patch.flatten(1, 2)   # [B, M*P', D_h]
        H_roi_flat  = H_roi.flatten(1, 2)      # [B, M*K', D_h]

        # Masks: use explicit mask from dataset (not derived from pixel_values)
        img_mask = self._get_img_mask(explicit_mask=image_mask)  # [B, M] — explicit 1/0

        # Build patch mask from image_mask: valid if image is real (any patch of real img)
        patch_mask = img_mask.unsqueeze(-1).expand(-1, -1, H_patch.size(2))  # [B, M, P]
        patch_mask_flat = patch_mask.reshape(B, -1)  # [B, M*P]

        # Build ROI mask from roi_seq BEFORE projector — roi_seq has true zeros for
        # padded slots (encode_roi fills them with torch.zeros). The projector has Linear
        # layers with bias, so H_roi = proj(roi_seq) is no longer zero for pad slots,
        # making any "sum != 0" check on H_roi unreliable for masking.
        roi_mask_pre = (roi_seq.sum(dim=-1) != 0).float()   # [B, M, K]
        roi_mask = roi_mask_pre.flatten(1, 2)                # [B, M*K]

        # Targets
        targets = self._get_targets(B, aspect_labels, device)  # [B, 6]

        # ── Step 2: Aspect loop ─────────────────────────────────────────────
        all_logits = []
        all_presence_logits = []
        all_queries = self.aspect_queries.all_queries()  # [6, D_h]

        for a in range(self.num_aspects):
            q_a = all_queries[a].unsqueeze(0).expand(B, -1)  # [B, D_h]

            # Retrieve evidence
            h_txt_a = self.text_retriever(q_a, H_txt, attention_mask)       # [B, D_h]
            h_img_a, _ = self.img_retriever(q_a, H_img_sum, img_mask)        # [B, D_h]
            h_patch_a, _ = self.patch_retriever(q_a, H_patch_flat, patch_mask_flat)  # [B, D_h]
            h_roi_a, _ = self.roi_retriever(q_a, H_roi_flat, roi_mask)     # [B, D_h]

            # Combine image summary + patch evidence (patch adds fine-grained detail)
            h_img_combined = h_img_a + 0.3 * h_patch_a

            # Fuse evidence
            h_fuse_a = self.gated_fusion(h_txt_a, h_img_combined, h_roi_a)  # [B, D_h]

            # Auxiliary presence logits
            presence_logits_a = self.presence_head(h_fuse_a).squeeze(-1)  # [B]
            all_presence_logits.append(presence_logits_a)

            # Build 5-token sequence
            seq_a, mask_a, pos_a = build_aspect_sequence(
                asp=q_a,
                txt_evi=h_txt_a,
                img_evi=h_img_combined,
                roi_evi=h_roi_a,
                fuse=h_fuse_a,
                device=device,
            )

            # Shared LLM decoder
            out_a = self.llm(
                inputs_embeds=seq_a,
                attention_mask=mask_a,
                position_ids=pos_a,
                use_cache=False,
            )
            h_last = out_a.last_hidden_state

            # Take last token (FUSE_a position)
            h_cls = h_last[:, -1, :]  # [B, D_h]
            logits_a = self.head(h_cls)  # [B, 4]
            all_logits.append(logits_a)

        logits = torch.stack(all_logits, dim=1)  # [B, 6, 4]
        presence_logits = torch.stack(all_presence_logits, dim=1)  # [B, 6]

        # ── Loss ─────────────────────────────────────────────────────────────
        loss = F.cross_entropy(
            logits.reshape(B * self.num_aspects, -1),
            targets.reshape(-1),
        )

        # Auxiliary presence loss: presence = 1 if class != Irrelative (0)
        presence_labels = (targets > 0).float()  # [B, 6]
        aux_loss = F.binary_cross_entropy_with_logits(
            presence_logits.reshape(-1),
            presence_labels.reshape(-1),
        )

        total_loss = loss + 0.1 * aux_loss

        return {
            "logits": logits.reshape(B * self.num_aspects, -1),
            "targets": targets.reshape(-1),
            "loss": total_loss,
            "ce_loss": loss,
            "aux_loss": aux_loss,
        }

    def inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        roi_data: List[List[Dict[str, Any]]],
        aspect_labels: List[Dict[int, int]],
        image_mask: torch.Tensor = None,
        raw_images: List[List[Any]] = None,
    ) -> torch.Tensor:
        """Inference mode — no gradient tracking."""
        with torch.no_grad():
            return self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                roi_data=roi_data,
                aspect_labels=aspect_labels,
                image_mask=image_mask,
                raw_images=raw_images,
            )["logits"]

    def get_trainable_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel() for name, p in self.named_parameters()
            if ("lora_A" in name or "lora_B" in name) and p.requires_grad
        )
        other_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "lora_A" not in name and "lora_B" not in name and p.requires_grad
        )
        return {"total": total, "lora": lora_params, "other": other_params}
