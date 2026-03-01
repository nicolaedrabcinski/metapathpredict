"""
Deep Reinforcement Learning for sequence classification.

Implements RL-based approach where an agent learns to classify
sequences by receiving rewards based on classification accuracy.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base import BaseModel
from .configurable_cnn import ConfigurableCNN


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor | None
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize buffer with given capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> list[Experience]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class SequenceEnvironment:
    """
    RL Environment for sequence classification.
    
    State: Encoded DNA sequence
    Action: Class prediction (0, 1, 2)
    Reward: +1 for correct, -1 for incorrect
    """
    
    def __init__(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor,
        reward_correct: float = 1.0,
        reward_incorrect: float = -0.5,
        reward_uncertain: float = -0.1,
    ):
        """
        Initialize environment.
        
        Args:
            sequences: All sequences [N, channels, length].
            labels: All labels [N].
            reward_correct: Reward for correct classification.
            reward_incorrect: Penalty for incorrect classification.
            reward_uncertain: Penalty for low-confidence predictions.
        """
        self.sequences = sequences
        self.labels = labels
        self.num_samples = len(sequences)
        
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.reward_uncertain = reward_uncertain
        
        self.current_idx = 0
        self.current_state = None
    
    def reset(self, idx: int | None = None) -> torch.Tensor:
        """
        Reset environment to new sequence.
        
        Args:
            idx: Specific index, or random if None.
        
        Returns:
            Initial state (sequence).
        """
        if idx is None:
            idx = random.randint(0, self.num_samples - 1)
        
        self.current_idx = idx
        self.current_state = self.sequences[idx]
        
        return self.current_state
    
    def step(self, action: int, confidence: float = 1.0) -> tuple[torch.Tensor | None, float, bool, dict]:
        """
        Take action (classify) and get reward.
        
        Args:
            action: Predicted class.
            confidence: Prediction confidence.
        
        Returns:
            (next_state, reward, done, info)
        """
        true_label = self.labels[self.current_idx].item()
        
        # Calculate reward
        if action == true_label:
            reward = self.reward_correct * confidence
        else:
            reward = self.reward_incorrect
            if confidence < 0.5:
                reward += self.reward_uncertain
        
        # Episode ends after one classification
        done = True
        next_state = None
        
        info = {
            "true_label": true_label,
            "predicted": action,
            "correct": action == true_label,
            "confidence": confidence,
        }
        
        return next_state, reward, done, info


class DQNAgent(BaseModel):
    """
    Deep Q-Network agent for sequence classification.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_actions: int = 3,
        backbone: str = "medium",
        hidden_dim: int = 256,
    ):
        """
        Initialize DQN agent.
        
        Args:
            in_channels: Input channels.
            num_actions: Number of actions (classes).
            backbone: CNN backbone preset.
            hidden_dim: Hidden dimension for Q-network.
        """
        super().__init__()
        
        # Feature extractor
        self.encoder = ConfigurableCNN(
            in_channels=in_channels,
            num_classes=num_actions,
            kernel_preset=backbone,
        )
        
        embed_dim = self.encoder._final_channels
        
        # Q-network head
        self.q_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )
        
        self.num_actions = num_actions
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for state.
        
        Args:
            state: State tensor [batch, channels, length].
        
        Returns:
            Q-values [batch, num_actions].
        """
        features = self.encoder.get_embeddings(state)
        q_values = self.q_network(features)
        return q_values
    
    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
    ) -> tuple[int, float]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            epsilon: Exploration rate.
        
        Returns:
            (action, confidence)
        """
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
            confidence = 1.0 / self.num_actions
        else:
            with torch.no_grad():
                if state.dim() == 2:
                    state = state.unsqueeze(0)
                
                q_values = self.forward(state)
                probs = F.softmax(q_values, dim=1)
                
                action = q_values.argmax(dim=1).item()
                confidence = probs[0, action].item()
        
        return action, confidence


class PolicyGradientAgent(BaseModel):
    """
    Policy Gradient (REINFORCE) agent for sequence classification.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_actions: int = 3,
        backbone: str = "medium",
        hidden_dim: int = 256,
    ):
        """
        Initialize policy gradient agent.
        
        Args:
            in_channels: Input channels.
            num_actions: Number of actions.
            backbone: CNN backbone preset.
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        
        self.encoder = ConfigurableCNN(
            in_channels=in_channels,
            num_classes=num_actions,
            kernel_preset=backbone,
        )
        
        embed_dim = self.encoder._final_channels
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=1),
        )
        
        # Value network (baseline)
        self.value = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        
        self.num_actions = num_actions
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get action probabilities and state value.
        
        Args:
            state: State tensor.
        
        Returns:
            (action_probs, state_value)
        """
        features = self.encoder.get_embeddings(state)
        action_probs = self.policy(features)
        state_value = self.value(features)
        return action_probs, state_value
    
    def select_action(self, state: torch.Tensor) -> tuple[int, float, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: Current state.
        
        Returns:
            (action, confidence, log_prob)
        """
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        probs, _ = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), probs[0, action].item(), log_prob


class ActorCriticAgent(BaseModel):
    """
    Actor-Critic agent combining policy and value learning.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_actions: int = 3,
        backbone: str = "medium",
        hidden_dim: int = 256,
    ):
        """Initialize actor-critic agent."""
        super().__init__()
        
        # Shared encoder
        self.encoder = ConfigurableCNN(
            in_channels=in_channels,
            num_classes=num_actions,
            kernel_preset=backbone,
        )
        
        embed_dim = self.encoder._final_channels
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        
        self.num_actions = num_actions
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get policy logits and state value."""
        features = self.encoder.get_embeddings(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, and value.
        
        Args:
            state: State tensor.
            action: Optional action to evaluate.
        
        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


class RLTrainer:
    """
    Trainer for RL-based sequence classification.
    """
    
    def __init__(
        self,
        agent: DQNAgent | PolicyGradientAgent | ActorCriticAgent,
        environment: SequenceEnvironment,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str = "cuda",
        gamma: float = 0.99,
        algorithm: str = "dqn",
    ):
        """
        Initialize RL trainer.
        
        Args:
            agent: RL agent.
            environment: Sequence environment.
            optimizer: Optimizer.
            device: Training device.
            gamma: Discount factor.
            algorithm: "dqn", "policy_gradient", or "actor_critic".
        """
        self.agent = agent.to(device)
        self.env = environment
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.algorithm = algorithm
        
        # For DQN
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.target_agent = None
        
        if algorithm == "dqn":
            # Create target network as exact copy of agent
            self.target_agent = copy.deepcopy(agent).to(device)
    
    def train_episode_dqn(
        self,
        batch_size: int = 32,
        epsilon: float = 0.1,
    ) -> dict[str, float]:
        """Train one episode using DQN."""
        state = self.env.reset().to(self.device)
        
        # Select and execute action
        action, confidence = self.agent.select_action(state.unsqueeze(0), epsilon)
        next_state, reward, done, info = self.env.step(action, confidence)
        
        # Store experience
        self.replay_buffer.push(Experience(
            state=state.cpu(),
            action=action,
            reward=reward,
            next_state=None,
            done=done,
        ))
        
        # Learn from replay buffer
        loss = 0.0
        if len(self.replay_buffer) >= batch_size:
            experiences = self.replay_buffer.sample(batch_size)
            loss = self._update_dqn(experiences)
        
        return {
            "reward": reward,
            "correct": float(info["correct"]),
            "loss": loss,
        }
    
    def _update_dqn(self, experiences: list[Experience]) -> float:
        """Update DQN from experiences."""
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], device=self.device)
        
        # Current Q-values
        q_values = self.agent(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (for terminal states, just reward)
        target_q_values = rewards
        
        # Loss
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode_policy_gradient(self) -> dict[str, float]:
        """Train one episode using REINFORCE."""
        state = self.env.reset().to(self.device)
        
        # Get action
        action, confidence, log_prob = self.agent.select_action(state.unsqueeze(0))
        _, reward, _, info = self.env.step(action, confidence)
        
        # Policy gradient loss
        _, baseline = self.agent(state.unsqueeze(0))
        advantage = reward - baseline.item()
        
        policy_loss = -log_prob * advantage
        value_loss = F.mse_loss(baseline.squeeze(), torch.tensor(reward, device=self.device))
        
        loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "reward": reward,
            "correct": float(info["correct"]),
            "loss": loss.item(),
        }
    
    def train_episode_actor_critic(self) -> dict[str, float]:
        """Train one episode using Actor-Critic."""
        state = self.env.reset().to(self.device).unsqueeze(0)
        
        # Get action and value
        action, log_prob, entropy, value = self.agent.get_action_and_value(state)
        _, reward, _, info = self.env.step(action.item())
        
        reward_tensor = torch.tensor([reward], device=self.device)
        
        # Advantage
        advantage = reward_tensor - value.detach()
        
        # Actor loss
        actor_loss = -(log_prob * advantage).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(value, reward_tensor)
        
        # Entropy bonus
        entropy_loss = -0.01 * entropy.mean()
        
        loss = actor_loss + 0.5 * critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "reward": reward,
            "correct": float(info["correct"]),
            "loss": loss.item(),
        }
    
    def train_epoch(
        self,
        num_episodes: int = 1000,
        epsilon: float = 0.1,
    ) -> dict[str, float]:
        """Train for multiple episodes."""
        total_reward = 0.0
        total_correct = 0
        total_loss = 0.0
        
        for _ in range(num_episodes):
            if self.algorithm == "dqn":
                result = self.train_episode_dqn(epsilon=epsilon)
            elif self.algorithm == "policy_gradient":
                result = self.train_episode_policy_gradient()
            else:
                result = self.train_episode_actor_critic()
            
            total_reward += result["reward"]
            total_correct += result["correct"]
            total_loss += result["loss"]
        
        return {
            "avg_reward": total_reward / num_episodes,
            "accuracy": total_correct / num_episodes,
            "avg_loss": total_loss / num_episodes,
        }
    
    def update_target_network(self) -> None:
        """Update target network (for DQN)."""
        if self.target_agent is not None:
            self.target_agent.load_state_dict(self.agent.state_dict())
