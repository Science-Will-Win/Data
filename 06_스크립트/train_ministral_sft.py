#!/usr/bin/env python3
"""Ministral 3B SFT Training - Biomedical QA fine-tuning with LoRA (no quantization)"""
import json, os, argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_PATH = "/home/sww/models/Ministral-3-3B-Instruct"
SFT_FILE = "phase2_handoff/sft_unified.jsonl"
OUTPUT_DIR = "ministral_sft_output"
SYSTEM_PROMPT = """You are a biomedical AI research agent. Given a research task, provide a precise answer based on biological databases and tools. Be concise and accurate."""

def load_sft_data():
    """Load SFT data and format as chat conversations"""
    data = []
    with open(SFT_FILE) as f:
        for line in f:
            d = json.loads(line)
            task_type = d.get("task_type", "unknown")
            benchmark = d.get("benchmark", "")
            answer = str(d.get("answer", ""))
            if not answer.strip():
                continue
            instruction = f"[{benchmark}/{task_type}] Solve this biomedical task and provide the answer."
            text = f"<s>[INST] {SYSTEM_PROMPT}\n\n{instruction}\n\nExpected format: Give a precise answer. [/INST] {answer}</s>"
            data.append({"text": text, "task_type": task_type})
    print(f"SFT data loaded: {len(data)} samples")
    from collections import Counter
    tc = Counter(d["task_type"] for d in data)
    for k, v in tc.most_common():
        print(f"  {k}: {v}")
    return Dataset.from_list(data)

def train(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = load_sft_data()
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3B model - no quantization needed, fits in single GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        
    )

    print(f"\n{'='*50}")
    print(f"Ministral 3B SFT Training")
    print(f"  Epochs: {args.epochs} | LR: {args.lr}")
    print(f"  Train: {len(split['train'])} | Val: {len(split['test'])}")
    print(f"  Effective batch: {2 * 4} = 8")
    print(f"{'='*50}\n")

    trainer.train()

    # Save
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    # Eval
    metrics = trainer.evaluate()
    print(f"\nFinal eval: {metrics}")
    with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
        json.dump({"eval_metrics": metrics, "epochs": args.epochs, "lr": args.lr}, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    train(p.parse_args())
