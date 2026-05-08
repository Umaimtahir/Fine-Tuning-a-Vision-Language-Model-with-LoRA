# Fine-Tuning-a-Vision-Language-Model-with-LoRA


# 🤖 Qwen2-VL Document-to-Markdown Generator
### Assignment 5 — AI4009 Generative AI | Spring 2026 | NUCES

Fine-tuning **Qwen2-VL-2B-Instruct** with LoRA for converting
document images into structured Markdown using the Nougat dataset.

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Model-umaimtahir1/qwen2vl--nougat--lora-blue)](https://huggingface.co/umaimtahir1/qwen2vl-nougat-lora)
[![Medium Blog](https://img.shields.io/badge/📝%20Blog-Medium-black)](https://medium.com/@umaimtahir)
[![Kaggle](https://img.shields.io/badge/💻%20Notebook-Kaggle-blue)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

---

## 📋 Assignment Overview

| Item | Detail |
|------|--------|
| Course | AI4009 Generative AI |
| Semester | Spring 2026 |
| University | NUCES |
| Task | Vision Language Model Fine-Tuning |
| Model | Qwen2-VL-2B-Instruct |
| Technique | LoRA (Parameter-Efficient Fine-Tuning) |
| Dataset | Nougat Training Dataset |
| Platform | Kaggle T4 × 2 GPU |

---

## 🎯 Objective

Convert document images (research papers, textbooks, lab reports)
into structured Markdown format using a fine-tuned Vision Language
Model. The model preserves headings, paragraphs, lists, tables,
and mathematical equations.

---

## 📁 Repository Structure


qwen2vl-document-markdown/
│
├── AI_ASS01_XXF_YYYY.ipynb     ← Main Kaggle notebook
├── lora_adapter_only/           ← Fine-tuned LoRA weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── outputs/
│   ├── loss_curves.png          ← Training & validation loss
│   ├── predictions.png          ← Input | GT | Generated
│   ├── rouge.png                ← ROUGE score bar chart
│   ├── unseen.png               ← Unseen document predictions
│   ├── epoch_comparison.png     ← Epoch 1 vs 2 vs 3 ROUGE
│   └── zeroshot_vs_finetuned.png
└── README.md

## 🏗️ Model Architecture
Document Image (512×512px)
│
▼
┌───────────────────┐
│  Visual Encoder   │  ← Qwen2-VL ViT (frozen)
│  (Patch Embedding)│
└────────┬──────────┘
│ Image embeddings
▼
┌───────────────────┐
│  Language Model   │  ← Qwen2-VL LM + LoRA adapters
│  + LoRA Adapters  │    Only 0.95% parameters trained
└────────┬──────────┘
│
▼
Markdown Output

---

## ⚙️ Training Configuration

```python
# LoRA Config
LORA_RANK     = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj","k_proj","v_proj","o_proj",
                 "gate_proj","up_proj","down_proj"]

# Training Config
BATCH_SIZE    = 2        # per GPU
GRAD_ACCUM    = 4        # effective batch = 8
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-4     # cosine decay
IMAGE_SIZE    = 512      # px
MAX_SEQ_LEN   = 768
```

---

## 📊 Results

### Training Progress

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|---------|--------|
| 1 | 0.2961 | 0.3748 | ✅ |
| 2 | 0.2244 | 0.3726 | ✅ 🏆 Best |
| 3 | ~0.180 | ~0.370 | ✅ |

### ROUGE Scores (Fine-Tuned vs Zero-Shot)

| Metric | Zero-Shot | Fine-Tuned | Δ |
|--------|-----------|-----------|---|
| ROUGE-1 | ~0.25 | ~0.50 | +100% |
| ROUGE-2 | ~0.10 | ~0.25 | +150% |
| ROUGE-L | ~0.22 | ~0.45 | +105% |

---

## 🚀 Quick Start

### Load the fine-tuned model

```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# Load base model + LoRA adapter
base = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(
    base, "umaimtahir1/qwen2vl-nougat-lora"
)
processor = AutoProcessor.from_pretrained(
    "umaimtahir1/qwen2vl-nougat-lora"
)
model.eval()

# Run inference
def generate_markdown(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",
         "text": "Convert this document image to Markdown."}
    ]}]

    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image],
                       return_tensors="pt").to("cuda")

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=512,
                             repetition_penalty=1.1)
    new = ids[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new[0], skip_special_tokens=True)

# Test it
result = generate_markdown("your_document.png")
print(result)
```

---

## 🛠️ Installation

```bash
git clone https://github.com/umaimtahir1/qwen2vl-document-markdown
cd qwen2vl-document-markdown
pip install transformers peft accelerate torch pillow gradio
```

---

## 📦 Dataset

**Nougat Training Dataset Example**
- Source: [Kaggle — zphilip/nougat-training-dataset-example](https://kaggle.com/datasets/zphilip/nougat-training-dataset-example)
- Format: Document page images + `.mmd` Markdown files
- Split: 80% train / 20% validation

---

## 🔑 Key Implementation Details

### ChatML Training Format
USER:      [document image] + instruction
ASSISTANT: # Generated Markdown output...
Labels are masked for the user turn — model trains only
on the assistant (Markdown) response.

### Corrupt Image Protection
All images pre-validated at dataset construction using
`PIL.Image.verify()` — corrupt files skipped automatically.

### Checkpoint Persistence
Checkpoints pushed to Kaggle dataset every 200 steps via
Kaggle API — survives the 12-hour session limit.

### Mixed Precision
`torch.amp.autocast("cuda")` + `GradScaler` activates
T4 tensor cores, pushing GPU utilization from 40% → 85%+.
