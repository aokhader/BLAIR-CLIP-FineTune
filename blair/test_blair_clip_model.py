import torch
from transformers import CLIPVisionConfig, CLIPVisionModel, RobertaConfig, RobertaModel

from blair.multimodal.blair_clip import BlairCLIPDualEncoder


def _build_test_model():
    text_config = RobertaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    vision_config = CLIPVisionConfig(
        hidden_size=32,
        projection_dim=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        image_size=32,
        patch_size=16,
    )
    text_encoder = RobertaModel(text_config)
    vision_model = CLIPVisionModel(vision_config)
    model = BlairCLIPDualEncoder(
        text_encoder=text_encoder,
        vision_model=vision_model,
        pooler_type="cls",
        projection_dim=16,
        logit_scale_init=0.07,
        text_temp=0.05,
        text_text_weight=0.5,
        do_mlm=False,
    )
    return model, vision_config


def test_blair_clip_forward_produces_loss_and_logits():
    model, vision_config = _build_test_model()
    model.train()
    batch_size = 3
    seq_len = 12
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, 2, seq_len))
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(batch_size, 3, vision_config.image_size, vision_config.image_size)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )

    assert outputs.loss is not None
    assert outputs.logits.shape == (batch_size, batch_size)


def test_encode_helpers_return_normalized_vectors():
    model, vision_config = _build_test_model()
    model.eval()

    text_ids = torch.randint(0, model.config.vocab_size, (1, 10))
    attention_mask = torch.ones_like(text_ids)
    text_embeds = model.encode_text(text_ids, attention_mask=attention_mask)
    assert torch.allclose(text_embeds.norm(dim=-1), torch.ones(1), atol=1e-5)

    image_values = torch.randn(1, 3, vision_config.image_size, vision_config.image_size)
    image_embeds = model.encode_image(image_values)
    assert torch.allclose(image_embeds.norm(dim=-1), torch.ones(1), atol=1e-5)

