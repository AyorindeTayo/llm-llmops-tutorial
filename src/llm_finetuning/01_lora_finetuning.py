"""
============================================================
MODULE 04 — LLM Fine-Tuning: LoRA & QLoRA
============================================================

WHAT YOU WILL LEARN:
  - What LoRA is and WHY it works mathematically
  - How to set up a LoRA fine-tuning job with HuggingFace PEFT
  - QLoRA: how 4-bit quantization makes it even more efficient
  - When to fine-tune vs when to use RAG

INTERVIEW QUESTIONS THIS COVERS:
  Q: What is LoRA?
  A: Low-Rank Adaptation. The idea: a weight update ΔW can be
     approximated as a product of two small matrices A and B,
     where rank r << d (dimensions). Instead of storing ΔW
     (d×d = millions of params), we store A and B (d×r + r×d).
     Example: d=4096, r=16 → 4096×4096=16.7M params vs
     4096×16 + 16×4096 = 131K params. 127x smaller!

  Q: What is QLoRA?
  A: Quantized LoRA. Load the base model in 4-bit (NF4 format),
     saving ~75% GPU memory, then apply LoRA on top in 16-bit
     precision. Allows fine-tuning a 70B model on a single A100.

  Q: What hyperparameters matter in LoRA?
  A: r (rank) — higher = more expressive but more params.
     alpha — scaling factor (usually alpha = 2×r).
     target_modules — which layers to adapt (q_proj, v_proj).
     lora_dropout — regularisation.

  Q: When fine-tune vs RAG?
  A: Fine-tune for: style, format, domain jargon, task-specific
     behaviour (e.g. always output JSON), reducing latency.
     RAG for: factual Q&A over documents, frequently updated info,
     when you can't retrain the model.
============================================================
"""

import os
from dataclasses import dataclass
from typing import Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    """
    All the hyperparameters you need to explain in an interview.
    """
    # Base model
    base_model: str = "meta-llama/Llama-2-7b-hf"
    # HF dataset
    dataset_name: str = "tatsu-lab/alpaca"

    # LoRA hyperparameters
    r: int = 16                  # Rank — lower = fewer params, higher = more expressive
    lora_alpha: int = 32         # Scaling factor: effective_lr = alpha/r
    lora_dropout: float = 0.05   # Dropout for regularisation
    target_modules: list = None  # Which layers get LoRA adapters

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 4×4 = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    output_dir: str = "./lora_output"

    def __post_init__(self):
        if self.target_modules is None:
            # Attention projection layers — standard LoRA targets
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class QLoRAConfig(LoRAConfig):
    """
    QLoRA adds 4-bit quantization to LoRA.
    """
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"  # Compute in fp16 even though stored in 4-bit
    bnb_4bit_quant_type: str = "nf4"         # NF4 = Normal Float 4-bit (better than int4)
    bnb_4bit_use_double_quant: bool = True   # Quantize the quantization constants too (saves more memory)


# ─────────────────────────────────────────────────────────────
# MATHEMATICAL EXPLANATION (no GPU needed)
# ─────────────────────────────────────────────────────────────

def explain_lora_math():
    """
    Explain LoRA with actual numpy numbers — great for interviews.
    """
    import numpy as np

    print("=" * 70)
    print("LoRA MATHEMATICAL EXPLANATION")
    print("=" * 70)

    d = 8   # Simplified: pretend model dimension is 8 (real: 4096+)
    r = 2   # Low rank

    # Original weight matrix W (frozen)
    W = np.random.randn(d, d)
    print(f"\n1. Original weight matrix W: shape {W.shape} = {d*d} parameters")

    # LoRA decomposition: ΔW ≈ B @ A
    A = np.random.randn(r, d)   # Shape: (r, d)
    B = np.zeros((d, r))        # Shape: (d, r) — initialized to ZERO so ΔW starts at 0

    print(f"\n2. LoRA matrices:")
    print(f"   A (down-projection): shape {A.shape} = {r*d} parameters")
    print(f"   B (up-projection):   shape {B.shape} = {d*r} parameters")
    print(f"   Total LoRA params:   {r*d + d*r} vs {d*d} original")
    print(f"   Compression ratio:   {d*d / (r*d + d*r):.1f}x fewer parameters!")

    # Forward pass with LoRA
    alpha = 4
    x = np.random.randn(d)

    output_original = W @ x
    delta_W = B @ A  # The weight update approximation
    output_lora = (W + (alpha / r) * delta_W) @ x

    print(f"\n3. Forward pass:")
    print(f"   Base model output:     {output_original[:3].round(3)}")
    print(f"   LoRA-adapted output:   {output_lora[:3].round(3)}")
    print(f"   Difference (adapter):  {(output_lora - output_original)[:3].round(3)}")
    print(f"\n   B is initialized to ZERO → at start, ΔW = B@A = 0")
    print(f"   → Training starts from original model behaviour (stable training!)")

    # Parameter counts for real models
    print("\n4. Real-world parameter comparison:")
    for dim, rank in [(4096, 8), (4096, 16), (4096, 64)]:
        full = dim * dim
        lora_params = rank * dim + dim * rank
        print(f"   d={dim}, r={rank}: Full={full/1e6:.1f}M, LoRA={lora_params/1000:.0f}K "
              f"({full/lora_params:.0f}x reduction)")


# ─────────────────────────────────────────────────────────────
# LORA TRAINING CODE (requires GPU + API key in production)
# ─────────────────────────────────────────────────────────────

def setup_lora_training(config: LoRAConfig, dry_run: bool = True):
    """
    Full LoRA fine-tuning setup using HuggingFace PEFT.

    Set dry_run=False and ensure HF_TOKEN is set to actually run.

    KEY PACKAGES:
      transformers — load base model and tokenizer
      peft         — add LoRA adapters
      trl          — SFTTrainer (Supervised Fine-Tuning)
      datasets     — load and format training data
    """
    if dry_run:
        print("\n[DRY RUN] LoRA training setup — set dry_run=False to execute")
        print(f"  Base model:   {config.base_model}")
        print(f"  LoRA rank:    {config.r}")
        print(f"  LoRA alpha:   {config.lora_alpha}")
        print(f"  Targets:      {config.target_modules}")
        print(f"  Epochs:       {config.num_train_epochs}")
        print(f"  Batch size:   {config.per_device_train_batch_size}")
        print(f"  Grad accum:   {config.gradient_accumulation_steps}")
        print(f"  Effective batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
        return

    # ── Real training code ──────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig as PEFTLoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import load_dataset

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA has no pad token by default

    # 2. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        token=os.getenv("HF_TOKEN"),
        device_map="auto",          # Automatically distribute across available GPUs
        torch_dtype="auto",
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # 3. Apply LoRA configuration
    peft_config = PEFTLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",  # Don't train bias terms
    )
    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable params: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")

    # 4. Load dataset
    dataset = load_dataset(config.dataset_name, split="train")

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="mlflow",   # Log to MLflow automatically
        fp16=True,
    )

    # 6. Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )
    trainer.train()

    # 7. Save adapter only (much smaller than full model)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"LoRA adapter saved to {config.output_dir}")


def setup_qlora_training(config: QLoRAConfig, dry_run: bool = True):
    """
    QLoRA extends LoRA with 4-bit quantization.
    Only the quantization config changes — everything else is the same.
    """
    if dry_run:
        print("\n[DRY RUN] QLoRA training setup — set dry_run=False to execute")
        print(f"  Quantization:    {config.bnb_4bit_quant_type} 4-bit")
        print(f"  Double quant:    {config.bnb_4bit_use_double_quant}")
        print(f"  Compute dtype:   {config.bnb_4bit_compute_dtype}")
        print(f"  LoRA rank:       {config.r}")
        print("\n  Memory savings vs full precision:")
        params_7b = 7_000_000_000
        fp16_gb = params_7b * 2 / 1e9
        int4_gb = params_7b * 0.5 / 1e9
        print(f"    7B model in fp16:  {fp16_gb:.0f} GB")
        print(f"    7B model in 4-bit: {int4_gb:.1f} GB  ({fp16_gb/int4_gb:.0f}x smaller!)")
        return

    from transformers import BitsAndBytesConfig, AutoModelForCausalLM
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # Then apply LoRA as in setup_lora_training above


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

def main():
    # 1. Show the math — no GPU needed
    explain_lora_math()

    # 2. Show LoRA setup
    print("\n" + "=" * 70)
    lora_config = LoRAConfig()
    setup_lora_training(lora_config, dry_run=True)

    # 3. Show QLoRA setup
    print("\n" + "=" * 70)
    qlora_config = QLoRAConfig()
    setup_qlora_training(qlora_config, dry_run=True)

    print("\n" + "=" * 70)
    print("KEY INTERVIEW POINTS — FINE-TUNING:")
    print("  LoRA:  Train only A and B matrices (low-rank), freeze base model")
    print("  QLoRA: Load base model in 4-bit + LoRA adapters in 16-bit")
    print("  RLHF:  Train reward model on human preferences → PPO to optimise LLM")
    print("  When to fine-tune: style/format changes, domain terminology, task alignment")
    print("  When to use RAG:   factual Q&A, frequently changing information")
    print("=" * 70)


if __name__ == "__main__":
    main()
