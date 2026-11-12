"""LoRA fine-tune Qwen2.5-1.5B Instruct teachers for BEVTSPD decision ranking."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(os.environ.get("ESNGS_QWEN_ROOT", str(Path(__file__).resolve().parents[4])))
sys.path.insert(0, str(ROOT / "pydeps"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_home"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "models" / "hf_home" / "transformers"))


class JsonlSFT(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = 2048):
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tok = tokenizer
        self.max_len = max_len
        self.examples = [self._encode(row, index) for index, row in enumerate(self.rows)]
        self.max_observed_len = max((example["sequence_length"] for example in self.examples), default=0)

    def __len__(self):
        return len(self.examples)

    def _encode(self, r, i):
        target = json.dumps(r["target"], ensure_ascii=False)
        prompt_messages = [
            {"role": "system", "content": "You output JSON only for BEVTSPD candidate ranking."},
            {"role": "user", "content": r["prompt"]},
        ]
        messages = prompt_messages + [{"role": "assistant", "content": target}]
        prompt_text = self.tok.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = self.tok(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]
        full_ids = self.tok(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > self.max_len:
            raise ValueError(
                f"full prompt/response length {len(full_ids)} exceeds --max-seq-len "
                f"{self.max_len} at row {i}"
            )
        enc = self.tok(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        # Response-only SFT: never train the model to reproduce system/user text.
        prompt_len = min(len(prompt_ids), int(attn.sum().item()))
        if input_ids[:prompt_len].tolist() != prompt_ids[:prompt_len]:
            raise ValueError(f"chat-template prefix mismatch at row {i}")
        labels[:prompt_len] = -100
        labels[attn == 0] = -100
        if not bool((labels != -100).any()):
            raise ValueError(
                f"assistant response truncated completely at row {i}; increase --max-seq-len"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "sequence_length": len(full_ids),
        }

    def __getitem__(self, i):
        example = self.examples[i]
        return {key: value for key, value in example.items() if key != "sequence_length"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = JsonlSFT(args.data_dir / "prompts_train.jsonl", tok, args.max_seq_len)
    val_ds = JsonlSFT(args.data_dir / "prompts_val.jsonl", tok, args.max_seq_len)
    print(
        f"n_train={len(train_ds)} n_val={len(val_ds)} max_seq={args.max_seq_len} "
        f"max_observed={max(train_ds.max_observed_len, val_ds.max_observed_len)}",
        flush=True,
    )

    targs = TrainingArguments(
        output_dir=str(args.out_dir / "hf_run"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        seed=args.seed,
        data_seed=args.seed,
    )
    progress = args.out_dir / "progress.log"
    progress.write_text("start\n", encoding="utf-8")

    from transformers import TrainerCallback

    class ProgressCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            line = f"step={state.global_step} {logs}\n"
            with progress.open("a", encoding="utf-8") as f:
                f.write(line)
            print(line, end="", flush=True)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=None,
        callbacks=[ProgressCB()],
    )
    trainer.train()
    adapter = args.out_dir / "adapter"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    meta = {
        "base_model": str(args.model_dir),
        "adapter": str(adapter),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "response_only_sft": True,
        "seed": args.seed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("DONE", meta)


if __name__ == "__main__":
    main()
