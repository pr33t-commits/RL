import matplotlib.pyplot as plt
import os

import torch


def plot_reward(reward_list, title):
    plt.plot(reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()


def save_training_checkpoint(
    checkpoint_path,
    model_state_dicts,
    optimizer_state_dicts=None,
    metadata=None,
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "models": model_state_dicts,
        "optimizers": optimizer_state_dicts or {},
        "metadata": metadata or {},
    }
    torch.save(checkpoint, checkpoint_path)


def load_training_checkpoint(
    checkpoint_path,
    models,
    optimizers=None,
    device="cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    for model_name, model in models.items():
        model.load_state_dict(checkpoint["models"][model_name])

    if optimizers is not None:
        for optimizer_name, optimizer in optimizers.items():
            if optimizer_name in checkpoint["optimizers"]:
                optimizer.load_state_dict(checkpoint["optimizers"][optimizer_name])

    return checkpoint.get("metadata", {})
