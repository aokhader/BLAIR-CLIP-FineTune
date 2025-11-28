import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from blair.simcse.models import MLPLayer, Pooler


class BlairCLIPDualEncoder(nn.Module):
    """
    Twin-tower encoder that pairs the existing BLaIR text tower with a CLIP vision tower.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        *,
        pooler_type: str = "cls",
        projection_dim: int = 512,
        clip_model_name: Optional[str] = None,
        vision_model: Optional[CLIPVisionModel] = None,
        logit_scale_init: float = 0.07,
        text_temp: float = 0.05,
        text_text_weight: float = 0.0,
        freeze_text: bool = False,
        freeze_vision: bool = False,
        mlp_only_train: bool = False,
        do_mlm: bool = False,
        mlm_weight: float = 0.1,
        cache_dir: Optional[str] = None,
        model_args: Optional[object] = None,
    ):
        super().__init__()
        if logit_scale_init <= 0:
            raise ValueError("logit_scale_init must be > 0.")
        if text_temp <= 0:
            raise ValueError("text_temp must be > 0.")

        self.text_encoder = text_encoder
        self.model_args = model_args
        self.config = text_encoder.config
        self.pooler_type = pooler_type
        self.pooler = Pooler(pooler_type)
        self.mlp_only_train = mlp_only_train
        self.projection_dim = projection_dim
        self.text_text_weight = text_text_weight
        self.do_mlm = do_mlm
        self.mlm_weight = mlm_weight

        if pooler_type == "cls":
            self.mlp = MLPLayer(self.config)
        else:
            self.mlp = None

        hidden_size = self.config.hidden_size
        self.text_projection = nn.Linear(hidden_size, projection_dim)

        if vision_model is not None:
            self.vision_model = vision_model
        elif clip_model_name is not None:
            self.vision_model = CLIPVisionModel.from_pretrained(clip_model_name, cache_dir=cache_dir)
        else:
            raise ValueError("Either `vision_model` or `clip_model_name` must be provided.")

        vision_hidden = getattr(self.vision_model.config, "hidden_size", None)
        if vision_hidden is None:
            raise ValueError("Unable to infer hidden size from the vision encoder.")
        self.image_projection = nn.Linear(vision_hidden, projection_dim)

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if freeze_vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / logit_scale_init)))
        text_temp_log = math.log(1.0 / text_temp)
        if text_text_weight > 0:
            self.text_text_logit_scale = nn.Parameter(torch.tensor(text_temp_log))
        else:
            self.register_buffer("text_text_logit_scale", torch.tensor(text_temp_log), persistent=False)

        if do_mlm:
            model_type = getattr(self.config, "model_type", "")
            if "roberta" in model_type:
                self.lm_head = RobertaLMHead(self.config)
            elif "bert" in model_type:
                self.lm_head = BertLMPredictionHead(self.config)
            else:
                raise ValueError("MLM head is only available for BERT/RoBERTa-style encoders.")

        self.cross_entropy = nn.CrossEntropyLoss()

    def resize_token_embeddings(self, new_num_tokens: int):
        if hasattr(self.text_encoder, "resize_token_embeddings"):
            self.text_encoder.resize_token_embeddings(new_num_tokens)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a batch of sentences (shape: [batch, seq_len]) into normalized embeddings.
        """
        if input_ids.dim() == 3:
            # Allow callers to pass the two-sentence format and default to the first sentence.
            input_ids = input_ids[:, 0]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 0]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, 0]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=self.pooler_type in ["avg_top2", "avg_first_last"],
            return_dict=True,
        )
        pooled = self.pooler(attention_mask, outputs)
        pooled = self._maybe_apply_mlp(pooled, force=False)
        projected = self.text_projection(pooled)
        return F.normalize(projected, dim=-1)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        pooled = outputs.pooler_output
        projected = self.image_projection(pooled)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sent_emb: bool = False,
        mlm_input_ids: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
    ):
        if return_dict is None:
            return_dict = True

        if sent_emb:
            embeddings = self.encode_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return BaseModelOutputWithPoolingAndCrossAttentions(pooler_output=embeddings)

        if pixel_values is None:
            raise ValueError("pixel_values must be provided for multimodal training.")

        text_embeddings = self._encode_text_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        text_anchor = text_embeddings[:, 0]
        meta_embeddings = text_embeddings[:, 1] if text_embeddings.size(1) > 1 else None

        image_embeddings = self.encode_image(pixel_values)

        gathered_text = self._gather_embeddings(text_anchor)
        gathered_images = self._gather_embeddings(image_embeddings)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_text = logit_scale * gathered_text @ gathered_images.t()
        logits_per_image = logits_per_text.t()
        labels = torch.arange(logits_per_text.size(0), device=logits_per_text.device)

        clip_loss = (
            self.cross_entropy(logits_per_text, labels) + self.cross_entropy(logits_per_image, labels)
        ) / 2.0
        loss = clip_loss

        if self.text_text_weight > 0 and meta_embeddings is not None:
            gathered_meta = self._gather_embeddings(meta_embeddings)
            text_text_scale = self.text_text_logit_scale.exp().clamp(max=100)
            text_text_logits = text_text_scale * gathered_text @ gathered_meta.t()
            text_text_loss = (
                self.cross_entropy(text_text_logits, labels)
                + self.cross_entropy(text_text_logits.t(), labels)
            ) / 2.0
            loss = loss + self.text_text_weight * text_text_loss

        if self.do_mlm and mlm_input_ids is not None and mlm_labels is not None:
            mlm_loss = self._compute_mlm_loss(
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
            )
            loss = loss + self.mlm_weight * mlm_loss

        if not return_dict:
            return (loss, logits_per_text)

        return SequenceClassifierOutput(loss=loss, logits=logits_per_text)

    def _encode_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        output_attentions: Optional[bool],
    ) -> torch.Tensor:
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.unsqueeze(1)

        batch_size, num_sent, seq_len = input_ids.size()
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        flat_token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None

        outputs = self.text_encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=self.pooler_type in ["avg_top2", "avg_first_last"],
            return_dict=True,
        )
        pooled = self.pooler(flat_attention_mask, outputs)
        pooled = pooled.view(batch_size, num_sent, -1)
        pooled = self._maybe_apply_mlp(pooled, force=True)
        projected = self.text_projection(pooled)
        return F.normalize(projected, dim=-1)

    def _maybe_apply_mlp(self, tensor: torch.Tensor, force: bool) -> torch.Tensor:
        if self.pooler_type != "cls" or self.mlp is None:
            return tensor
        if force or self.training or not self.mlp_only_train:
            return self.mlp(tensor)
        return tensor

    def _gather_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized() or not self.training:
            return embeddings
        tensors = [torch.zeros_like(embeddings) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors, embeddings.contiguous())
        tensors[dist.get_rank()] = embeddings
        return torch.cat(tensors, dim=0)

    def _compute_mlm_loss(
        self,
        mlm_input_ids: torch.Tensor,
        mlm_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        output_attentions: Optional[bool],
    ) -> torch.Tensor:
        _, _, seq_len = mlm_input_ids.size()
        flat_ids = mlm_input_ids.view(-1, seq_len)
        flat_labels = mlm_labels.view(-1, seq_len)
        flat_attention = attention_mask.view(-1, seq_len)
        flat_token_type = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None

        outputs = self.text_encoder(
            input_ids=flat_ids,
            attention_mask=flat_attention,
            token_type_ids=flat_token_type,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )

        prediction_scores = self.lm_head(outputs.last_hidden_state)
        vocab_size = prediction_scores.size(-1)
        return self.cross_entropy(
            prediction_scores.view(-1, vocab_size),
            flat_labels.view(-1),
        )

