#!/usr/bin/env python3
"""Preference Reward Model training on local GSM8K solution rollouts.

For each problem in the JSONL file, every correct rollout is paired with every
incorrect rollout. The model is trained with Bradley--Terry loss:
``-log(sigmoid(r_chosen - r_rejected))``.

Usage:
    python -m reward_models.train_preference_rm_gsm8k \
        --config reward_models/configs/preference_rm_gsm8k.yaml
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from reward_models.base import (
    BaseRewardModel,
    create_optimizer,
    finish_wandb,
    init_wandb,
    load_tokenizer,
    log_metrics,
    pad_sequences,
)
from reward_models.config import Config, load_config


# =============================================================================
# Data preparation
# =============================================================================


def tokenize_math_completion(
    tokenizer: AutoTokenizer, prompt: str, completion: str, max_length: int
) -> Dict[str, List[int]]:
    """Tokenize prompt and rollout, retaining the rollout's final answer.

    Correctness in GSM8K is expressed at the end of a rollout. When a sequence
    exceeds ``max_length``, retain all of the prompt when possible and truncate
    the *start* of the completion, rather than accidentally removing its final
    answer as ordinary right truncation would.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        completion + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    if len(prompt_ids) + len(completion_ids) > max_length:
        if len(prompt_ids) >= max_length:
            prompt_ids = prompt_ids[: max_length // 2]
        completion_budget = max_length - len(prompt_ids)
        completion_ids = completion_ids[-completion_budget:]
    input_ids = prompt_ids + completion_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

def build_preference_dataset(tokenizer: AutoTokenizer, config: Config) -> Dataset:
    """Create all correct-vs-incorrect rollout pairs within every problem.

    ``dataset_name`` is a path to a JSONL rollout file. Expected fields are
    ``problem_index`` (problem ID), ``question``, ``solution``, and ``label``;
    a label of 1 is correct and 0 is incorrect. For a problem with C correct
    and I incorrect rollouts, this produces C * I preference pairs.
    """
    rollout_path = Path(config.dataset_name)
    if not rollout_path.is_file():
        raise FileNotFoundError(f"Rollout JSONL file not found: {rollout_path}")

    grouped: dict[str, dict[str, object]] = {}
    with rollout_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"problem_index", "question", "solution", "label"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"Line {line_number} is missing required fields: {sorted(missing)}")
            problem_id = str(row["problem_index"])
            group = grouped.setdefault(
                problem_id, {"question": row["question"].strip(), "correct": [], "incorrect": []}
            )
            if group["question"] != row["question"].strip():
                raise ValueError(f"Problem {problem_id} has inconsistent question text.")
            bucket = "correct" if int(row["label"]) == 1 else "incorrect"
            group[bucket].append(row["solution"].strip())

    problem_ids = list(grouped)
    random.Random(config.seed).shuffle(problem_ids)
    problem_ids = problem_ids[: min(config.samples, len(problem_ids))]
    records = []
    skipped_problems = dropped_identical = 0
    for problem_id in problem_ids:
        group = grouped[problem_id]
        correct = group["correct"]
        incorrect = group["incorrect"]
        if not correct or not incorrect:
            skipped_problems += 1
            continue
        prompt = f"Question: {group['question']}\nAnswer:"
        chosen_rollouts = [tokenize_math_completion(tokenizer, prompt, solution, config.max_length) for solution in correct]
        rejected_rollouts = [tokenize_math_completion(tokenizer, prompt, solution, config.max_length) for solution in incorrect]
        for chosen in chosen_rollouts:
            for rejected in rejected_rollouts:
                if chosen["input_ids"] == rejected["input_ids"]:
                    dropped_identical += 1
                    continue
                records.append(
                    {
                        "problem_id": problem_id,
                        "chosen_ids": chosen["input_ids"],
                        "chosen_mask": chosen["attention_mask"],
                        "rejected_ids": rejected["input_ids"],
                        "rejected_mask": rejected["attention_mask"],
                    }
                )

    print(f"Selected {len(problem_ids)} problems; skipped {skipped_problems} without both labels.")
    if dropped_identical:
        print(f"Dropped {dropped_identical} pairs identical after max_length truncation.")
    return Dataset.from_list(records)


def split_by_problem(data: Dataset, val_ratio: float, seed: int) -> tuple[Dataset, Dataset | None]:
    """Split pairs by problem ID to prevent rollout leakage into validation."""
    if val_ratio == 0.0:
        return data, None
    problem_ids = sorted(set(data["problem_id"]))
    if len(problem_ids) < 2:
        print("Fewer than two usable problems; training without a validation split.")
        return data, None
    random.Random(seed).shuffle(problem_ids)
    val_problem_count = min(len(problem_ids) - 1, max(1, round(len(problem_ids) * val_ratio)))
    validation_ids = set(problem_ids[:val_problem_count])
    train_indices = [index for index, row in enumerate(data) if row["problem_id"] not in validation_ids]
    val_indices = [index for index, row in enumerate(data) if row["problem_id"] in validation_ids]
    return data.select(train_indices), data.select(val_indices)


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """Pad the chosen and rejected sequence in every preference pair."""
    return {
        "chosen_ids": pad_sequences([item["chosen_ids"] for item in batch], tokenizer.pad_token_id),
        "chosen_mask": pad_sequences([item["chosen_mask"] for item in batch], 0),
        "rejected_ids": pad_sequences([item["rejected_ids"] for item in batch], tokenizer.pad_token_id),
        "rejected_mask": pad_sequences([item["rejected_mask"] for item in batch], 0),
    }


# =============================================================================
# Model definition
# =============================================================================


class PreferenceRewardModel(BaseRewardModel):
    """A scalar, sequence-level reward head on the ORM-compatible backbone."""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, head_dim=1, **kwargs)

    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the head value at each sequence's last non-padding token."""
        hidden = self.get_hidden_states(input_ids, attention_mask)
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        return self.head(hidden[batch_indices, last_indices]).squeeze(-1)

    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r_chosen = self.get_reward(chosen_ids, chosen_mask)
        r_rejected = self.get_reward(rejected_ids, rejected_mask)
        return -F.logsigmoid(r_chosen - r_rejected).mean(), r_chosen, r_rejected


# =============================================================================
# Evaluation, artifacts, and training
# =============================================================================


@torch.no_grad()
def evaluate_preference_rm(
    model: PreferenceRewardModel,
    loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_device_type: str,
) -> dict[str, float]:
    model.eval()
    loss_total = correct_total = pair_total = 0
    margin_total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.amp.autocast(autocast_device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
            loss, chosen, rejected = model(**batch)
        pairs = chosen.numel()
        loss_total += loss.item() * pairs
        correct_total += (chosen > rejected).sum().item()
        margin_total += (chosen - rejected).sum().item()
        pair_total += pairs
    count = max(1, pair_total)
    return {
        "val/loss": loss_total / count,
        "val/accuracy": correct_total / count,
        "val/reward_margin": margin_total / count,
    }


def save_model_artifacts(
    model: PreferenceRewardModel,
    tokenizer: AutoTokenizer,
    config: Config,
    history: dict[str, list],
) -> Path:
    """Save checkpoint, tokenizer, configuration, and tracked training loss."""
    output_dir = Path(__file__).resolve().parent / "trained_models" / "preference_rm_gsm8k"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_state.pt")
    torch.save(model.head.state_dict(), output_dir / "reward_head.pt")
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(config.model_dump_json(indent=2) + "\n")
    (output_dir / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["train"]["steps"], history["train"]["loss"], label="train loss")
        if history["validation"]["loss"]:
            ax.plot(history["validation"]["steps"], history["validation"]["loss"], label="validation loss")
        ax.set(xlabel="Optimizer step", ylabel="Loss", title="Preference RM Training Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("matplotlib is not installed; skipping loss-curve plot.")
    print(f"Saved trained preference-RM artifacts to {output_dir}")
    return output_dir


def train_preference_rm_gsm8k(config: Config) -> PreferenceRewardModel:
    """Train, validate, log, and save a GSM8K Bradley--Terry reward model."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.get_device())
    init_wandb("preference_rm_gsm8k", config.model_dump(), config.use_wandb)
    tokenizer = load_tokenizer(config.model_id)

    print(f"Building rollout preference dataset from {config.dataset_name}...")
    data = build_preference_dataset(tokenizer, config)
    if len(data) == 0:
        raise ValueError("No usable pairs were produced. Increase max_length or samples.")
    print(f"Dataset size: {len(data)} preference pairs")
    train_data, val_data = split_by_problem(data, config.val_ratio, config.seed)
    print(f"Train size: {len(train_data)} pairs")
    if val_data is not None:
        print(f"Validation size: {len(val_data)} pairs")

    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                        drop_last=len(train_data) > config.batch_size,
                        collate_fn=lambda batch: collate_fn(batch, tokenizer))
    val_loader = (DataLoader(val_data, batch_size=config.batch_size, shuffle=False,
                             collate_fn=lambda batch: collate_fn(batch, tokenizer))
                  if val_data is not None else None)

    print(f"Loading model: {config.model_id}")
    model = PreferenceRewardModel(config.model_id, freeze_backbone=config.freeze_backbone, device=device).to(device)
    print(f"Trainable parameters: {model.count_trainable_params() / 1e6:.2f}M")
    optimizer = create_optimizer(model, config.lr)
    total_steps = -(-len(loader) // config.grad_accum_steps) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = None
    if config.lr_scheduler == "linear_decay":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif config.lr_scheduler == "warmup_only" and warmup_steps:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

    autocast_enabled = device.type == "cuda" and torch.cuda.is_available()
    autocast_device_type = "cuda" if autocast_enabled else "cpu"
    history = {"train": {"steps": [], "loss": [], "accuracy": [], "margin": []},
               "validation": {"steps": [], "loss": [], "accuracy": [], "margin": []}}
    global_step = 0
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        model.train()
        epoch_loss = epoch_correct = epoch_pairs = 0
        accum_loss = accum_correct = accum_pairs = 0
        accum_margin = 0.0
        accum_microbatches = 0
        optimizer.zero_grad()

        for step, batch in tqdm(enumerate(loader), total=len(loader)):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(autocast_device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, chosen, rejected = model(**batch)
            (loss / config.grad_accum_steps).backward()
            correct = (chosen > rejected).sum().item()
            pairs = chosen.numel()
            margin = (chosen - rejected).sum().item()
            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_pairs += pairs
            accum_loss += loss.item()
            accum_correct += correct
            accum_pairs += pairs
            accum_margin += margin
            accum_microbatches += 1

            if (step + 1) % config.grad_accum_steps == 0 or step + 1 == len(loader):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                avg_loss = accum_loss / accum_microbatches
                accuracy = accum_correct / max(1, accum_pairs)
                avg_margin = accum_margin / max(1, accum_pairs)
                print(f"Epoch {epoch + 1} step {global_step} | loss {avg_loss:.4f} | acc {accuracy:.3f} | margin {avg_margin:.4f}")
                history["train"]["steps"].append(global_step)
                history["train"]["loss"].append(avg_loss)
                history["train"]["accuracy"].append(accuracy)
                history["train"]["margin"].append(avg_margin)
                log_metrics({"train/loss": avg_loss, "train/accuracy": accuracy,
                             "train/reward_margin": avg_margin}, step=global_step)
                accum_loss = accum_correct = accum_pairs = 0
                accum_margin = 0.0
                accum_microbatches = 0

                if val_loader is not None and config.eval_interval > 0 and global_step % config.eval_interval == 0:
                    metrics = evaluate_preference_rm(model, val_loader, device, autocast_enabled, autocast_device_type)
                    print(f"Eval step {global_step} | val loss {metrics['val/loss']:.4f} | val acc {metrics['val/accuracy']:.3f}")
                    for key, value in (("loss", metrics["val/loss"]), ("accuracy", metrics["val/accuracy"]), ("margin", metrics["val/reward_margin"])):
                        history["validation"][key].append(value)
                    history["validation"]["steps"].append(global_step)
                    log_metrics(metrics, step=global_step)
                    model.train()

        print(f"Epoch {epoch + 1} | loss {epoch_loss / len(loader):.4f} | accuracy {epoch_correct / max(1, epoch_pairs):.3f}")

        if val_loader is not None and (config.eval_interval <= 0 or global_step % config.eval_interval != 0):
            metrics = evaluate_preference_rm(model, val_loader, device, autocast_enabled, autocast_device_type)
            print(f"Epoch {epoch + 1} | val loss {metrics['val/loss']:.4f} | val acc {metrics['val/accuracy']:.3f}")
            history["validation"]["steps"].append(global_step)
            history["validation"]["loss"].append(metrics["val/loss"])
            history["validation"]["accuracy"].append(metrics["val/accuracy"])
            history["validation"]["margin"].append(metrics["val/reward_margin"])
            log_metrics(metrics, step=global_step)
            model.train()

    save_model_artifacts(model, tokenizer, config, history)
    finish_wandb()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GSM8K preference reward model")
    parser.add_argument("--config", required=True, help="Path to a GSM8K preference YAML config")
    args = parser.parse_args()
    train_preference_rm_gsm8k(load_config(args.config))


if __name__ == "__main__":
    main()
