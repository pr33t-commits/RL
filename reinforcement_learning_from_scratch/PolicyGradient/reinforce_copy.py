from dataclasses import dataclass, field
from typing import List, Tuple
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from ple import PLE
from ple.games.pixelcopter import Pixelcopter
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass
class ReinforceConfig:
    env_name: str = "Pixelcopter-PLE-v0"
    seed: int = 0
    hidden_dim: int = 640
    learning_rate: float = 1e-4
    gamma: float = 0.99
    temperature_start: float = 1.0
    temperature_end: float = 1.0
    max_episodes: int = 4000
    eval_interval: int = 100
    eval_episodes: int = 30
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


def make_env(config: ReinforceConfig) -> gym.Env:
    if config.env_name in {"Pixelcopter-PLE-v0", "PixelCopter-v0"}:
        return PixelcopterEnv(seed=config.seed, render_mode="human" if config.render else None)
    return gym.make(config.env_name, render_mode="human" if config.render else None)

@dataclass
class EpisodeBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    action_log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.action_log_probs.clear()
        self.rewards.clear()

    def add(self, state: np.ndarray, action: int, action_log_probs: torch.Tensor, reward: float) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.action_log_probs.append(action_log_probs)
        self.rewards.append(float(reward))

    def __len__(self) -> int:
        return len(self.states)

class ReinforceAgent:
    def __init__(self, env: gym.Env, config: ReinforceConfig) -> None:
        # Store the environment, config, and target device.
        # Read the observation and action dimensions from the environment.
        # Build the policy network and optimizer here.
        # Create any episode buffer / logging containers you want to reuse.
        self.env = env
        self.statedim = env.observation_space.shape[0]
        self.actiondim = env.action_space.n
        self.config = config
        self.policynet = PolicyNet(self.statedim, self.config.hidden_dim, self.actiondim).to(self.config.device)
        self.buffer = EpisodeBuffer()
        self.optimizer = torch.optim.Adam(self.policynet.parameters(), lr=self.config.learning_rate)
        self.train_rewards: List[float] = []
        self.eval_checkpoints: List[int] = []
        self.eval_rewards: List[float] = []
        self.eval_rewards_stds: List[float] = []
        self.temperature_schedule_episodes = int(min(1500, 0.5 * self.config.max_episodes))
        # raise NotImplementedError("Initialize the REINFORCE agent here.")

    def get_temperature(self, episode: int) -> float:
        if self.temperature_schedule_episodes <= 0:
            return self.config.temperature_end
        progress = min(episode / self.temperature_schedule_episodes, 1.0)
        return self.config.temperature_start + progress * (
            self.config.temperature_end - self.config.temperature_start
        )

    def select_action(self, state: np.ndarray, temperature: float) -> Tuple[int, float]:
        """
        Return sampled action and its log probability.

        Replace this with your own action-sampling logic if you want to derive
        the policy step fully from scratch.
        """
        # Convert the state into a tensor on the correct device.
        # Pass it through the policy network to get action scores or probabilities.
        # Create a categorical distribution for CartPole's discrete action space.
        # Sample an action and compute its log probability.
        # Return both the chosen action and the log probability.
        
        action_logits = self.policynet.forward(torch.tensor(state, dtype=torch.float32, device=self.config.device))
        temperature = max(temperature, 1e-6)
        action_probs = torch.softmax(action_logits / temperature, dim=-1)
        # action = torch.multinomial(action_probs, num_samples=1).item()
        action = torch.distributions.Categorical(action_probs).sample().item()
        
        return action, torch.log(action_probs[action])
        # raise NotImplementedError("Implement action selection here.")
        

    def compute_returns(self) -> torch.Tensor:
        """
        Compute discounted returns for one complete trajectory.

        Replace this if you want to write the Monte Carlo return calculation
        yourself.
        """
        # Walk backward through the rewards collected in the current episode.
        # Maintain a running discounted return G_t = r_t + gamma * G_{t+1}.
        # Reverse the collected values back into episode order.
        # Convert the returns into a tensor on the correct device.
        # Optionally normalize the returns for training stability.
        # raise NotImplementedError("Implement return computation here.")
        G = []
        rewards = self.buffer.rewards
        for i in range(len(rewards)-1, -1, -1):
            if i == len(rewards)-1:
                G.insert(0,rewards[i])
            else:
                G.insert(0,rewards[i] + (self.config.gamma * G[0]))
        G_t = torch.tensor(G, dtype=torch.float32, device=self.config.device)
        eps = np.finfo(np.float32).eps.item()
        G_t = (G_t - G_t.mean()) / (G_t.std() + eps)
        return G_t

    def update(self) -> None:
        """
        Run the REINFORCE policy-gradient update for the collected episode.

        This structure is intentionally straightforward so you can swap in your
        own return computation, baseline, or loss expression.
        """
        # Exit early if the episode buffer is empty.
        # Convert buffered states and actions into tensors.
        # Call compute_returns() to get the Monte Carlo targets.
        # Recompute log probabilities for the taken actions from the current policy.
        # Build the REINFORCE loss: maximize return-weighted log probability.
        # Zero gradients, backpropagate, and step the optimizer.
        # Clear the episode buffer once the update is done.
        # raise NotImplementedError("Implement the REINFORCE update here.")
        if len(self.buffer.states) == 0:
            return
        G = self.compute_returns()
        # states = torch.concat([torch.tensor(s, dtype=torch.float32, device=self.config.device) for s in self.buffer.states], dim = 0)
        log_probs_tensor = torch.stack(self.buffer.action_log_probs)
        loss = - (log_probs_tensor * G).sum() #/ len(G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer.clear()

    def train(self) -> List[float]:
        # Loop for the configured number of episodes.
        # Reset the environment at the start of each episode and clear the buffer.
        # Repeatedly:
        # 1. choose an action with select_action()
        # 2. step the environment
        # 3. store state, action, and reward in the episode buffer
        # 4. accumulate episode reward
        # Stop when the episode terminates or truncates.
        # After the episode ends, log the reward and call update().
        # Optionally print moving averages every few episodes.
        # Return the list of training rewards.
        for episode in range(self.config.max_episodes):
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{self.config.max_episodes}")
            temperature = self.get_temperature(episode)
            state, _ = self.env.reset(seed=self.config.seed + episode)
            self.buffer.clear()
            terminated = False
            truncated = False
            episode_reward = 0.0
            while (not terminated) and (not truncated):
                action, log_prob = self.select_action(state, temperature)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.buffer.add(state, action, log_prob, reward)
                state = next_state
                episode_reward += reward
            
            self.update()
            self.train_rewards.append(episode_reward)
            
            

            if (episode + 1) % self.config.eval_interval == 0:
                evaluation_rewards = self.evaluate(self.config.eval_episodes)
                mean_eval_reward = float(np.mean(evaluation_rewards))
                self.eval_checkpoints.append(episode + 1)
                self.eval_rewards.append(mean_eval_reward)
                self.eval_rewards_stds.append(float(np.std(evaluation_rewards)))
                print(f"Actions: {self.buffer.actions}")
                print(f"Evaluation after episode {episode + 1}: {mean_eval_reward:.2f} ± {self.eval_rewards_stds[-1]:.2f}")

        return self.train_rewards

    def evaluate(self, num_episodes: int) -> List[float]:
        # Create a fresh evaluation environment, optionally with rendering enabled.
        # For each evaluation episode:
        # 1. reset the environment
        # 2. repeatedly choose actions from the current policy
        # 3. accumulate rewards until done / truncated
        # Store each episode return in a list.
        # Close the evaluation environment before returning.
        # raise NotImplementedError("Implement evaluation here.")
        cum_rewards = []
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset(seed=self.config.seed + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0
            while (not terminated) and (not truncated):
                action, log_prob = self.select_action(state, self.config.temperature_end)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                episode_reward += reward
            cum_rewards.append(episode_reward)
        return cum_rewards

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    config = ReinforceConfig()
    set_seed(config.seed)

    env = make_env(config)
    agent = ReinforceAgent(env=env, config=config)

    training_rewards = agent.train()

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Average training reward: {np.mean(training_rewards):.2f}")
    if agent.eval_rewards:
        print(f"Average evaluation reward: {agent.eval_rewards[-1]:.2f}")

    output_dir = f"Results_reinforce_{config.env_name}"
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot training reward on primary axis
    ax1.plot(training_rewards, label="Training Reward", alpha=0.7)
    ax1.plot(agent.eval_checkpoints, agent.eval_rewards, label="Eval Reward (Mean)", marker='o', color='orange')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Plot eval_rewards_stds on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(agent.eval_checkpoints, agent.eval_rewards_stds, label="Eval Reward (Std Dev)", marker='s', color='red', alpha=0.7)
    ax2.set_ylabel("Standard Deviation", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title("REINFORCE Training and Evaluation Rewards")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_eval_rewards.png"))
    plt.close()
    
    env.close()


if __name__ == "__main__":
    main()
