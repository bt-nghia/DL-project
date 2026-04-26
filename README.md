# Full Fine-Tuning (FFT) of RoBERTa and LLaMA on GLUE & SuperGLUE

A self-contained guide explaining the input, preprocessing, model setup, training, and evaluation pipeline for full fine-tuning on NLU benchmarks.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Benchmarks](#2-benchmarks)
3. [Models](#3-models)
4. [Input & Preprocessing](#4-input--preprocessing)
5. [Model Setup & Full Fine-Tuning](#5-model-setup--full-fine-tuning)
6. [Training](#6-training)
7. [Output & Evaluation](#7-output--evaluation)
8. [Results](#8-results)
9. [Submission](#9-submission)

---

## 1. Overview

**Full Fine-Tuning (FFT)** trains *all* model parameters on a downstream task — unlike parameter-efficient methods (LoRA, adapters) which freeze most of the backbone. FFT typically yields the strongest results but requires significantly more compute.

```
Raw Text → Tokenizer → Model (all params trainable) → Logits → Loss → Backprop → Predictions → Metric
```

| Property | RoBERTa | LLaMA |
|---|---|---|
| Architecture | Encoder-only (BERT-style) | Decoder-only (GPT-style) |
| Parameters | 125M (base) / 355M (large) | 7B / 13B / 70B |
| Classification head | Linear on `[CLS]` token | Linear on last non-padding token |
| Typical use case | NLU tasks (GLUE, SuperGLUE) | NLU + generation tasks |

---

## 2. Benchmarks

### GLUE (General Language Understanding Evaluation)

| Task | Type | Labels | Metric |
|---|---|---|---|
| SST-2 | Sentiment classification | 2 (Pos/Neg) | Accuracy |
| QQP | Question pair similarity | 2 (Dup/Not) | F1 + Accuracy |
| CoLA | Linguistic acceptability | 2 | Matthews Corr. |
| STS-B | Semantic textual similarity | Float [0–5] | Pearson / Spearman |
| MRPC | Paraphrase detection | 2 | F1 + Accuracy |
| RTE | Textual entailment | 2 | Accuracy |

### SuperGLUE

| Task | Type | Labels | Metric |
|---|---|---|---|
| BoolQ | Reading comprehension | 2 (True/False) | Accuracy |
| CB | Commitmentbank NLI | 3 | Accuracy + F1 |
| MultiRC | Multi-sentence QA | Binary per candidate | F1a + Exact Match |
| WiC | Word-in-context | 2 | Accuracy |
| RTE | Textual entailment | 2 | Accuracy |

---

## 3. Models

### RoBERTa

RoBERTa (Robustly Optimized BERT Pretraining Approach) is a BERT variant pretrained with dynamic masking, no next-sentence prediction, and larger batches.

- **Tokenizer:** Byte-Pair Encoding (BPE), vocabulary size 50k
- **Special tokens:** `<s>` (CLS), `</s>` (SEP/EOS)
- **Pair separator:** `</s></s>` between sentence A and B
- **Max length:** 512 tokens

### LLaMA 2

LLaMA 3.2 is a decoder-only autoregressive model pretrained on 2T tokens.

- **Tokenizer:** SentencePiece BPE, vocabulary size 32k
- **Special tokens:** `<s>` (BOS), `</s>` (EOS)
- **No `[CLS]` token** — classification uses the last non-padding token's hidden state
- **Architecture details:** Grouped Query Attention (GQA), SwiGLU FFN, RoPE positional embeddings

---

## 4. Input & Preprocessing

### Step 1 — Load dataset

```python
from datasets import load_dataset

# GLUE task
dataset = load_dataset("glue", "sst2")

# SuperGLUE task
dataset = load_dataset("super_glue", "boolq")
```

Each dataset returns `train`, `validation`, and `test` splits.


### Step 2 — Tokenize

#### RoBERTa tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Single sentence
encoded = tokenizer(
    "great film",
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
# → input_ids:      [0, 8343, 1012, 5, 2, 1, 1, ...]
# → attention_mask: [1,    1,    1, 1, 1, 0, 0, ...]

# Sentence pair
encoded = tokenizer(
    premise, hypothesis,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
```

**Resulting token structure for pairs:**

```
<s> sentence_A </s> </s> sentence_B </s> [PAD] ...
```

#### LLaMA tokenization

```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token  # LLaMA has no default pad token

encoded = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

> **Note:** For LLaMA, a prompt template is often prepended to help the model understand the task:
> ```
> "Classify the sentiment of the following text.\nText: {sentence}\nSentiment:"
> ```

### Step 4 — Create DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=32,          # 4–8 for LLaMA
    shuffle=True
)
```

Each batch contains tensors of shape `[batch_size, max_length]`:
- `input_ids`
- `attention_mask`
- `labels` — shape `[batch_size]`

---

## 5. Model Setup & Full Fine-Tuning

### RoBERTa — classification head on `[CLS]`

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2       # adjust per task
)

# Full fine-tuning: ALL parameters are trainable
for param in model.parameters():
    param.requires_grad = True

print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
# → 124,647,170 (roberta-base)
```

**Forward pass:**

```
input_ids  →  Token Embeddings (vocab × 768)
           →  12× Transformer Block
                 └─ Multi-head Self-Attention
                 └─ Feed-Forward Network (768 → 3072 → 768)
                 └─ LayerNorm + Residual
           →  [CLS] hidden state  (shape: [batch, 768])
           →  Dense(768, 768) + Tanh
           →  Dropout
           →  Dense(768, num_labels)
           →  logits              (shape: [batch, num_labels])
```

### LLaMA — classification head on last token

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    num_labels=2,
    torch_dtype=torch.bfloat16,     # required for memory
    device_map="auto"               # multi-GPU sharding
)
model.config.pad_token_id = model.config.eos_token_id

# Full fine-tuning: ALL parameters are trainable
for param in model.parameters():
    param.requires_grad = True

print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
# → ~6,738,415,616 (Llama-2-7b)
```

**Forward pass:**

```
input_ids  →  Token Embeddings (32000 × 4096)
           →  32× Transformer Block
                 └─ Grouped Query Attention (GQA) + RoPE
                 └─ SwiGLU Feed-Forward (4096 → 11008 → 4096)
                 └─ RMSNorm + Residual
           →  Last non-padding token hidden state  (shape: [batch, 4096])
           →  Linear(4096, num_labels)
           →  logits                               (shape: [batch, num_labels])
```

> **Key difference from RoBERTa:** There is no `[CLS]` token. The model uses the last meaningful token's representation for classification.

---

## 6. Training

### Optimizer and scheduler

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",

    # RoBERTa settings
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=5,

    # LLaMA settings (override above)
    # learning_rate=1e-5,
    # per_device_train_batch_size=4,
    # gradient_accumulation_steps=8,
    # num_train_epochs=3,

    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,              # gradient clipping

    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    fp16=True,                      # RoBERTa
    # bf16=True,                    # LLaMA
    # gradient_checkpointing=True,  # LLaMA (reduces memory)
)
```

### Hyperparameter summary

| Hyperparameter | RoBERTa-base | LLaMA-2-7B |
|---|---|---|
| Learning rate | 2e-5 | 1e-5 |
| Batch size | 32 | 4–8 |
| Gradient accumulation | 1 | 8 |
| Epochs | 3–5 | 2–3 |
| LR warmup | 6% of steps | 3% of steps |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | 1.0 | 1.0 |
| Precision | fp16 | bf16 |
| Gradient checkpointing | No | Yes |
| Approx. VRAM needed | ~8 GB | ~56 GB (bf16) |

### Loss computation

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# During training loop
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits                    # shape: [batch, num_labels]
loss = criterion(logits, labels)           # scalar
loss.backward()
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

---

## 7. Output & Evaluation

### What the model outputs

```python
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

outputs.logits
# tensor([[ 2.31, -1.08]])   shape: [batch_size, num_labels]
# Raw unnormalized scores — one per class

predicted_class = outputs.logits.argmax(dim=-1)
# tensor([0])   → class index → map to label string
# e.g. {0: "Positive", 1: "Negative"} for SST-2

probabilities = outputs.logits.softmax(dim=-1)
# tensor([[0.974, 0.026]])
```

### Computing metrics

```python
import evaluate

# Choose metric per task
metric = evaluate.load("glue", "sst2")       # accuracy
# metric = evaluate.load("glue", "qqp")      # f1 + accuracy
# metric = evaluate.load("glue", "cola")     # matthews_correlation

metric.add_batch(
    predictions=predicted_class.cpu().numpy(),
    references=labels.cpu().numpy()
)
result = metric.compute()
# {"accuracy": 0.948}
```

### Evaluation loop

```python
model.eval()
for batch in eval_loader:
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

result = metric.compute()
print(result)
```


## 8. Submission
Running Scripts and code is included in the repository for example running Llama on GLUE dataset script: sh run_llama_glue.sh --seed 42 --dataset mrpc
