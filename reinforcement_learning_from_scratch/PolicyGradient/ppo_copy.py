from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    env_name: str = "CartPole-v1"
    seed: int = 0
    hidden_dim: int = 128
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 10
    rollout_steps: int = 2048
    minibatch_size: int = 256
    max_training_steps: int = 50_000
    eval_episodes: int = 10
    render: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value


@dataclass
class RolloutBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    def __init__(self, env: gym.Env, config: PPOConfig) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(config.device)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.actor = PolicyNet(self.state_dim, config.hidden_dim, self.action_dim).to(self.device)
        self.critic = ValueNet(self.state_dim, config.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.buffer = RolloutBuffer()
        self.train_rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Return sampled action, its old log probability, and the critic value.

        Populate this with your PPO action-sampling logic.
        """
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.actor(state_tensor)
            value = self.critic(state_tensor)

        action_probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def compute_advantages_and_returns(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and value targets for the collected rollout.

        Replace this implementation if you want to derive it yourself.
        """
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)

        if len(self.buffer.next_states) == 0:
            raise ValueError("Rollout buffer is empty.")

        final_next_state = torch.tensor(
            [self.buffer.next_states[-1]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            last_value = self.critic(final_next_state).item()

        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for step in reversed(range(len(rewards))):
            next_value = last_value if step == len(rewards) - 1 else values[step + 1]
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * next_value * mask - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[step] = gae
            returns[step] = gae + values[step]

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        return advantages_tensor, returns_tensor

    def prepare_batch_tensors(self) -> Dict[str, torch.Tensor]:
        states = torch.tensor(np.asarray(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.int64, device=self.device).view(-1, 1)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device).view(-1, 1)
        advantages, returns = self.compute_advantages_and_returns()

        return {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": advantages.view(-1, 1),
            "returns": returns.view(-1, 1),
        }

    def update(self) -> None:
        """
        Run PPO updates using the rollout buffer.

        This method is intentionally structured so you can swap in your own
        ratio, clipped objective, entropy bonus, and critic-loss formulation.
        """
        if len(self.buffer) == 0:
            return

        batch = self.prepare_batch_tensors()
        num_samples = batch["states"].shape[0]

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(num_samples, device=self.device)

            for start in range(0, num_samples, self.config.minibatch_size):
                batch_indices = indices[start:start + self.config.minibatch_size]
                states = batch["states"][batch_indices]
                actions = batch["actions"][batch_indices]
                old_log_probs = batch["old_log_probs"][batch_indices]
                advantages = batch["advantages"][batch_indices]
                returns = batch["returns"][batch_indices]

                logits = self.actor(states)
                action_probs = torch.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(actions.squeeze(-1)).view(-1, 1)
                entropy = action_dist.entropy().mean()
                state_values = self.critic(states)

                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped_objective = ratio * advantages
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                )
                clipped_objective = clipped_ratio * advantages

                actor_loss = -torch.min(unclipped_objective, clipped_objective).mean()
                critic_loss = F.mse_loss(state_values, returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.buffer.clear()

    def train(self) -> List[float]:
        state, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0.0
        total_steps = 0

        while total_steps < self.config.max_training_steps:
            action, log_prob, value = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
            )

            episode_reward += reward
            total_steps += 1
            state = next_state

            if done:
                self.train_rewards.append(episode_reward)
                if len(self.train_rewards) % 10 == 0:
                    print(
                        f"Episode {len(self.train_rewards):4d} | "
                        f"Average reward (last 10): {np.mean(self.train_rewards[-10:]):.2f}"
                    )
                state, _ = self.env.reset()
                episode_reward = 0.0

            if len(self.buffer) >= self.config.rollout_steps:
                self.update()

        if len(self.buffer) > 0:
            self.update()

        return self.train_rewards

    def evaluate(self, num_episodes: int) -> List[float]:
        rewards = []
        eval_env = gym.make(
            self.config.env_name,
            render_mode="human" if self.config.render else None,
        )

        for episode_idx in range(num_episodes):
            state, _ = eval_env.reset(seed=self.config.seed + episode_idx)
            done = False
            truncated = False
            episode_reward = 0.0

            while not done and not truncated:
                action, _, _ = self.select_action(state)
                state, reward, done, truncated, _ = eval_env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)

        eval_env.close()
        return rewards


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    config = PPOConfig()
    set_seed(config.seed)

    env = gym.make(config.env_name)
    agent = PPOAgent(env=env, config=config)

    training_rewards = agent.train()
    evaluation_rewards = agent.evaluate(config.eval_episodes)

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Average evaluation reward: {np.mean(evaluation_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
