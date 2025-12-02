# BLaIR

BLaIR, which is short for "**B**ridging **La**nguage and **I**tems for **R**etrieval and **R**ecommendation", is a series of language models pre-trained on Amazon Reviews 2023 dataset. [[📑 Paper](https://arxiv.org/abs/2403.03952)]

**🤗 Checkpoints:**
* [blair-roberta-base](https://huggingface.co/hyp1231/blair-roberta-base) (125M)
* [blair-roberta-large](https://huggingface.co/hyp1231/blair-roberta-large) (355M)

## Architecture

The first two released checkpoints ([blair-roberta-base](https://huggingface.co/hyp1231/blair-roberta-base) and [blair-roberta-large](https://huggingface.co/hyp1231/blair-roberta-large)) follow the same model architecture of [RoBERTa](https://arxiv.org/abs/1907.11692).

## Use

### Download Checkpoints

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
model = AutoModel.from_pretrained("hyp1231/blair-roberta-base")
```

### Get Embeddings

```python
language_context = 'I need a product that can scoop, measure, and rinse grains without the need for multiple utensils and dishes. It would be great if the product has measurements inside and the ability to rinse and drain all in one. I just have to be careful not to pour too much accidentally.'
item_metadata = [
  'Talisman Designs 2-in-1 Measure Rinse & Strain | Holds up to 2 Cups | Food Strainer | Fruit Washing Basket | Strainer & Colander for Kitchen Sink | Dishwasher Safe - Dark Blue. The Measure Rinse & Strain by Talisman Designs is a 2-in-1 kitchen colander and strainer that will measure and rinse up to two cups. Great for any type of food from rice, grains, beans, fruit, vegetables, pasta and more. After measuring, fill with water and swirl to clean. Strain then pour into your pot, pan, or dish. The convenient size is easy to hold with one hand and is compact to fit into a kitchen cabinet or pantry. Dishwasher safe and food safe.',
  'FREETOO Airsoft Gloves Men Tactical Gloves for Hiking Cycling Climbing Outdoor Camping Sports (Not Support Screen Touch).'
]
texts = [language_context] + item_metadata

inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    embeddings = model(**inputs, return_dict=True).last_hidden_state[:, 0]
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
```

### Compare Similarities

```python
print(embeddings[0] @ embeddings[1])    # tensor(0.8564)
print(embeddings[0] @ embeddings[2])    # tensor(0.5741)
```

## Training

<center>
    <img src="../assets/blair.png" style="width: 75%;">
</center>

BLaIR is grounded on pairs of *(item metadata, language context)*, enabling the models to:
* derive strong item text representations, for both recommendation and retrieval;
* predict the most relevant item given simple / complex language context.

[blair-roberta-base](https://huggingface.co/hyp1231/blair-roberta-base) is initialized by the parameters of [roberta-base](https://huggingface.co/FacebookAI/roberta-base). Then the model was continually pretrained on 10% of all the <review, item metadata> pairs from the [Amazon Reviews 2023 dataset](https://github.com/hyp1231/AmazonReviews2023).

### Requirements

```
pytorch==2.1.1
pytorch-cuda==11.8
python==3.9.18
transformers==4.2.1
```

### Prepare your pretraining data

```bash
cd blair/
python sample_pretraining_data.py
```

Note that the required datasets will be automatically downloaded from [huggingface dataset hub](https://github.com/hyp1231/AmazonReviews2023).

For the multimodal variant (which also stores local image paths), run:

```bash
cd blair/
python sample_multimodal_data.py \
  --categories Appliances \
  --image_dir blair_clip_images \
  --output_tsv clean_review_meta_with_images.tsv
```

### Train your models

**blair-roberta-base**

```bash
bash base.sh
```

**blair-roberta-large**

```bash
bash large.sh
```

Our training script heavily referenced the open-source implementation of [SimCSE](https://github.com/princeton-nlp/SimCSE). We adapt it to an environment with relatively modern versions of PyTorch & HuggingFace Transformers.

### Multimodal BLaIR + CLIP

We now support a twin-tower configuration that adds a CLIP vision encoder next to the original text tower. Enable it via:

```bash
python train.py \
  --model_family blair_clip \
  --model_name_or_path roberta-base \
  --mm_clip_model_name openai/clip-vit-base-patch16 \
  --mm_projection_dim 512 \
  --mm_text_text_weight 0.5 \
  --image_column image_path \
  --image_root /path/to/images \
  --train_file clean_review_meta_with_images.tsv \
  --output_dir checkpoints/blair-mm \
  --per_device_train_batch_size 64 \
  --num_train_epochs 1 \
  --do_train --do_eval
```

Where `image_column` points to a column in the training TSV/CSV that stores image paths (either absolute or relative to `image_root`). Additional knobs:

- `--mm_temperature_init`: initial CLIP temperature τ (default `0.07`).
- `--mm_freeze_clip_tower` / `--mm_freeze_text_tower`: freeze either tower for efficient fine-tuning.
- `--mm_text_text_weight`: optionally keep the original text-text contrastive objective in addition to the text–image loss.

The multimodal pipeline reuses the optimizer/scheduler infrastructure from the existing training script and supports distributed execution via `torchrun` in the same way as the text-only models.
