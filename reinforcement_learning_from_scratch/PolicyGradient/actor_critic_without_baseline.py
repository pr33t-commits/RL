from dataclasses import dataclass
from typing import List
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from ple import PLE
from ple.games.pixelcopter import Pixelcopter
from tqdm import tqdm
from util import load_training_checkpoint, save_training_checkpoint

@dataclass
class ActorCriticConfig:
    env_name: str = "Pixelcopter-PLE-v0"
    seed: int = 0
    actor_hidden_dim: int = 240
    q_hidden_dim: int = 360
    actor_lr: float = 1e-4
    q_lr: float = 1e-4
    gamma: float = 0.99
    max_episodes: int = 3000
    eval_interval: int = 100
    eval_episodes: int = 30
    checkpoint_dir: str = "checkpoints_actor_critic"
    checkpoint_name: str = "actor_critic_latest.pt"
    resume_training: bool = False
    render: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, 12)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        # logits_int = self.fc2(x)
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)

class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PixelcopterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, seed: int = 0, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.game = Pixelcopter()
        self.game_state = PLE(
            self.game,
            fps=30,
            display_screen=render_mode == "human",
            state_preprocessor=None,
        )
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self._action_set))
        self._seed_rng(seed)

    def _seed_rng(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = rng

    def _extract_state(self) -> np.ndarray:
        game_state = self.game.getGameState()
        return np.array(
            [
                game_state["player_y"],
                game_state["player_vel"],
                game_state["player_dist_to_floor"],
                game_state["player_dist_to_ceil"],
                game_state["next_gate_dist_to_player"],
                game_state["next_gate_block_top"],
                game_state["next_gate_block_bottom"],
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_rng(seed)
        self.game_state.reset_game()
        return self._extract_state(), {}

    def step(self, action: int):
        reward = self.game_state.act(self._action_set[action])
        observation = self._extract_state()
        terminated = self.game_state.game_over()
        truncated = False
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        if hasattr(self.game_state, "display_screen"):
            self.game_state.display_screen = False


def make_env(config: ActorCriticConfig) -> gym.Env:
    if config.env_name in {"Pixelcopter-PLE-v0", "PixelCopter-v0"}:
        return PixelcopterEnv(seed=config.seed, render_mode="human" if config.render else None)
    return gym.make(config.env_name, render_mode="human" if config.render else None)


class ActorCriticAgent:
    def __init__(self, env: gym.Env, config: ActorCriticConfig) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.actor = PolicyNet(self.state_dim, config.actor_hidden_dim, self.action_dim).to(self.device)
        self.q_net = QNet(self.state_dim, self.action_dim, config.q_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.q_lr)

        self.train_rewards: List[float] = []
        self.eval_checkpoints: List[int] = []
        self.eval_rewards: List[float] = []
        self.eval_rewards_stds: List[float] = []
        self.completed_episodes = 0

    def take_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        probs = self.actor(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def action_to_tensor(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action.view(-1), num_classes=self.action_dim).float()

    def update_td(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor([action], dtype=torch.int64, device=self.device)
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([[done]], dtype=torch.float32, device=self.device)

        state_action_value = self.q_net(state_tensor, self.action_to_tensor(action_tensor))

        with torch.no_grad():
            next_action_probs = self.actor(next_state_tensor)
            next_action_dist = torch.distributions.Categorical(next_action_probs)
            next_action_tensor = next_action_dist.sample()
            next_action_value = self.q_net(
                next_state_tensor,
                self.action_to_tensor(next_action_tensor),
            )
            td_target = reward_tensor + self.config.gamma * (1 - done_tensor) * next_action_value

        td_error = td_target - state_action_value

        action_probs = self.actor(state_tensor)
        selected_action_probs = action_probs.gather(1, action_tensor.view(-1, 1))
        log_probs = torch.log(selected_action_probs.clamp_min(1e-8))
        actor_loss = torch.mean(-log_probs * state_action_value.detach())
        q_loss = torch.mean(td_error.pow(2))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

    def save_checkpoint(self, checkpoint_path: str) -> None:
        save_training_checkpoint(
            checkpoint_path=checkpoint_path,
            model_state_dicts={
                "actor": self.actor.state_dict(),
                "q_net": self.q_net.state_dict(),
            },
            optimizer_state_dicts={
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
            },
            metadata={
                "completed_episodes": self.completed_episodes,
                "train_rewards": self.train_rewards,
                "eval_checkpoints": self.eval_checkpoints,
                "eval_rewards": self.eval_rewards,
                "eval_rewards_stds": self.eval_rewards_stds,
            },
        )

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        metadata = load_training_checkpoint(
            checkpoint_path=checkpoint_path,
            models={
                "actor": self.actor,
                "q_net": self.q_net,
            },
            optimizers={
                "actor_optimizer": self.actor_optimizer,
                "q_optimizer": self.q_optimizer,
            },
            device=self.device,
        )
        self.completed_episodes = metadata.get("completed_episodes", 0)
        self.train_rewards = metadata.get("train_rewards", [])
        self.eval_checkpoints = metadata.get("eval_checkpoints", [])
        self.eval_rewards = metadata.get("eval_rewards", [])
        self.eval_rewards_stds = metadata.get("eval_rewards_stds", [])
        return metadata

    def train(self, num_episodes: int | None = None, checkpoint_path: str | None = None) -> List[float]:
        if num_episodes is None:
            num_episodes = self.config.max_episodes

        start_episode = self.completed_episodes
        end_episode = start_episode + num_episodes

        for episode in range(start_episode, end_episode):
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{end_episode}")

            state, _ = self.env.reset(seed=self.config.seed + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0
            actions = []

            while not terminated and not truncated:
                action = self.take_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                actions.append(action)
                done = terminated or truncated

                self.update_td(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

            self.train_rewards.append(episode_reward)
            self.completed_episodes = episode + 1

            if (episode + 1) % self.config.eval_interval == 0:
                evaluation_rewards = self.evaluate(self.config.eval_episodes)
                mean_eval_reward = float(np.mean(evaluation_rewards))
                self.eval_checkpoints.append(episode + 1)
                self.eval_rewards.append(mean_eval_reward)
                self.eval_rewards_stds.append(float(np.std(evaluation_rewards)))
                print(
                    f"Evaluation after episode {episode + 1}: "
                    f"{mean_eval_reward:.2f} +/- {self.eval_rewards_stds[-1]:.2f}"
                )
                print(actions)
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)

        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)

        return self.train_rewards

    def evaluate(self, num_episodes: int) -> List[float]:
        rewards = []
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset(seed=self.config.seed + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated:
                action = self.take_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)

        return rewards

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def main() -> None:
    config = ActorCriticConfig()
    set_seed(config.seed)

    env = make_env(config)
    agent = ActorCriticAgent(env=env, config=config)
    checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)

    if config.resume_training and os.path.exists(checkpoint_path):
        try:
            agent.load_checkpoint(checkpoint_path)
            print(
                f"Loaded checkpoint from episode {agent.completed_episodes}. "
                "Continuing with TD Q-value updates."
            )
        except RuntimeError:
            print(
                "Existing checkpoint is incompatible with the new Q-network architecture. "
                "Starting training from scratch."
            )
        training_rewards = agent.train(
            num_episodes=config.max_episodes,
            checkpoint_path=checkpoint_path,
        )
    else:
        training_rewards = agent.train(
            num_episodes=config.max_episodes,
            checkpoint_path=checkpoint_path,
        )

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Average training reward: {np.mean(training_rewards):.2f}")
    if agent.eval_rewards:
        print(f"Average evaluation reward: {agent.eval_rewards[-1]:.2f}")

    output_dir = f"Results_actor_critic_no_baseline_{config.env_name}"
    os.makedirs(output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(training_rewards, label="Training Reward", alpha=0.7)
    ax1.plot(agent.eval_checkpoints, agent.eval_rewards, label="Eval Reward (Mean)", marker="o", color="orange")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()
    ax2.plot(
        agent.eval_checkpoints,
        agent.eval_rewards_stds,
        label="Eval Reward (Std Dev)",
        marker="s",
        color="red",
        alpha=0.7,
    )
    ax2.set_ylabel("Standard Deviation", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Actor-Critic Training and Evaluation Rewards")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_eval_rewards.png"))
    plt.close()

    env.close()

if __name__ == "__main__":
    main()
