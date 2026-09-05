"""
Fine-tunes Llama-3 8B with QLoRA (4-bit quantized base + low-rank
adapters) using Direct Preference Optimization on the dataset produced by
prepare_dpo_dataset.py. Requires the 'training' extras:

    pip install -e ".[training]"

Usage:
    python training/train_qlora_dpo.py \
        --dataset training/output/dpo_dataset.jsonl \
        --output-dir training/output/dpo_adapter
"""

import argparse

from src.config.settings import get_settings


def main() -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="training/output/dpo_adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    base_model_id = get_settings().local_base_model_id

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=quant_config, device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model, args=dpo_config, train_dataset=dataset, tokenizer=tokenizer, peft_config=lora_config
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved DPO-tuned LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()