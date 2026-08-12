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
    actor_hidden_dim: int = 120
    critic_hidden_dim: int = 480
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    method = 'td'
    gamma: float = 0.99
    max_episodes: int = 3000
    eval_interval: int = 100
    eval_episodes: int = 30
    checkpoint_dir: str = "checkpoints_actor_critic"
    checkpoint_name: str = "actor_critic_latest.pt"
    resume_training: bool = True
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

class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.critic = ValueNet(self.state_dim, config.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.train_rewards: List[float] = []
        self.eval_checkpoints: List[int] = []
        self.eval_rewards: List[float] = []
        self.eval_rewards_stds: List[float] = []
        self.completed_episodes = 0

    def take_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        probs = self.actor(state_tensor)
        # print(probs)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update_td(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([[done]], dtype=torch.float32, device=self.device)

        td_target = reward_tensor + self.config.gamma * (1 - done_tensor) * self.critic(next_state_tensor)
        td_delta = td_target - self.critic(state_tensor)

        log_probs = torch.log(self.actor(state_tensor).gather(1, action_tensor))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = F.mse_loss(self.critic(state_tensor), td_target.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        returns = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + self.config.gamma * running_return
            returns.append(running_return)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=self.device).view(-1, 1)

    def update_monte_carlo(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
    ) -> None:
        state_tensor = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).view(-1, 1)
        returns = self.compute_returns(rewards)

        state_values = self.critic(state_tensor)
        advantages = returns - state_values
        log_probs = torch.log(self.actor(state_tensor).gather(1, action_tensor))

        actor_loss = torch.mean(-log_probs * advantages.detach())
        critic_loss = F.mse_loss(state_values, returns.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save_checkpoint(self, checkpoint_path: str, method: str) -> None:
        save_training_checkpoint(
            checkpoint_path=checkpoint_path,
            model_state_dicts={
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            optimizer_state_dicts={
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            metadata={
                "completed_episodes": self.completed_episodes,
                "train_rewards": self.train_rewards,
                "eval_checkpoints": self.eval_checkpoints,
                "eval_rewards": self.eval_rewards,
                "eval_rewards_stds": self.eval_rewards_stds,
                "method": method,
            },
        )

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        metadata = load_training_checkpoint(
            checkpoint_path=checkpoint_path,
            models={
                "actor": self.actor,
                "critic": self.critic,
            },
            optimizers={
                "actor_optimizer": self.actor_optimizer,
                "critic_optimizer": self.critic_optimizer,
            },
            device=self.device,
        )
        self.completed_episodes = metadata.get("completed_episodes", 0)
        self.train_rewards = metadata.get("train_rewards", [])
        self.eval_checkpoints = metadata.get("eval_checkpoints", [])
        self.eval_rewards = metadata.get("eval_rewards", [])
        self.eval_rewards_stds = metadata.get("eval_rewards_stds", [])
        return metadata

    def train(self, method: str = "td", num_episodes: int | None = None, checkpoint_path: str | None = None) -> List[float]:
        method_name = method.lower()
        if method_name not in {"td", "monte_carlo", "mc"}:
            raise ValueError("method must be one of: 'td', 'monte_carlo', 'mc'")
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
            states = []
            rewards = []

            while not terminated and not truncated:
                action = self.take_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                actions.append(action)
                states.append(state)
                rewards.append(reward)
                done = terminated or truncated

                if method_name == "td":
                    self.update_td(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

            if method_name in {"monte_carlo", "mc"}:
                self.update_monte_carlo(states, actions, rewards)

            self.train_rewards.append(episode_reward)
            self.completed_episodes = episode + 1

            if (episode + 1) % self.config.eval_interval == 0:
                print(f"Actions taken {actions}")
                evaluation_rewards = self.evaluate(self.config.eval_episodes)
                mean_eval_reward = float(np.mean(evaluation_rewards))
                self.eval_checkpoints.append(episode + 1)
                self.eval_rewards.append(mean_eval_reward)
                self.eval_rewards_stds.append(float(np.std(evaluation_rewards)))
                print(
                    f"Evaluation after episode {episode + 1}: "
                    f"{mean_eval_reward:.2f} +/- {self.eval_rewards_stds[-1]:.2f}"
                )
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path, method_name)

        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path, method_name)

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
        metadata = agent.load_checkpoint(checkpoint_path)
        checkpoint_method = metadata.get("method", config.method)
        print(
            f"Loaded checkpoint from episode {agent.completed_episodes}. "
            f"Continuing with method '{checkpoint_method}'."
        )
        training_rewards = agent.train(
            method=checkpoint_method,
            num_episodes=config.max_episodes,
            checkpoint_path=checkpoint_path,
        )
    else:
        training_rewards = agent.train(
            method=config.method,
            num_episodes=config.max_episodes,
            checkpoint_path=checkpoint_path,
        )

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Average training reward: {np.mean(training_rewards):.2f}")
    if agent.eval_rewards:
        print(f"Average evaluation reward: {agent.eval_rewards[-1]:.2f}")

    output_dir = f"Results_actor_critic_{config.env_name}"
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
