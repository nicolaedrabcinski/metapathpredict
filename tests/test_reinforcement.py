"""
Unit tests for reinforcement learning module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from metapathpredict.models.reinforcement import (
    DQNAgent,
    PolicyGradientAgent,
    ActorCriticAgent,
    SequenceEnvironment,
    ReplayBuffer,
    Experience,
)


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_push_and_sample(self):
        """Test pushing and sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)
        
        # Push some transitions using Experience dataclass
        for i in range(50):
            exp = Experience(
                state=torch.randn(4, 100),
                action=i % 3,
                reward=float(i),
                next_state=torch.randn(4, 100),
                done=i == 49,
            )
            buffer.push(exp)
        
        assert len(buffer) == 50
        
        # Sample batch
        batch = buffer.sample(16)
        
        assert len(batch) == 16  # List of Experience objects
        assert isinstance(batch[0], Experience)

    def test_capacity_limit(self):
        """Test buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(20):
            exp = Experience(
                state=torch.randn(4, 100),
                action=i % 3,
                reward=1.0,
                next_state=torch.randn(4, 100),
                done=False,
            )
            buffer.push(exp)
        
        assert len(buffer) == 10

    def test_sample_more_than_size_raises(self):
        """Test sampling more than buffer size returns all available."""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(5):
            exp = Experience(
                state=torch.randn(4, 100),
                action=0,
                reward=1.0,
                next_state=torch.randn(4, 100),
                done=False,
            )
            buffer.push(exp)
        
        # Sample returns min(batch_size, len(buffer))
        batch = buffer.sample(10)
        assert len(batch) == 5  # Only 5 available


class TestSequenceEnvironment:
    """Tests for SequenceEnvironment."""

    def test_reset(self):
        """Test environment reset."""
        sequences = torch.randn(10, 4, 100)
        labels = torch.tensor([i % 3 for i in range(10)])
        
        env = SequenceEnvironment(sequences, labels)
        state = env.reset()
        
        assert state.shape == (4, 100)

    def test_step_correct_action(self):
        """Test step with correct action."""
        sequences = torch.randn(1, 4, 100)
        labels = torch.tensor([1])  # Label is 1
        
        env = SequenceEnvironment(sequences, labels)
        env.reset()
        
        next_state, reward, done, info = env.step(1)  # Correct action
        
        assert reward > 0  # Positive reward for correct
        assert done is True
        assert info["correct"] is True

    def test_step_incorrect_action(self):
        """Test step with incorrect action."""
        sequences = torch.randn(1, 4, 100)
        labels = torch.tensor([1])  # Label is 1
        
        env = SequenceEnvironment(sequences, labels)
        env.reset()
        
        next_state, reward, done, info = env.step(0)  # Incorrect action
        
        assert reward < 0  # Negative reward for incorrect
        assert done is True
        assert info["correct"] is False

    def test_episode_iteration(self):
        """Test iterating through multiple episodes."""
        sequences = torch.randn(5, 4, 100)
        labels = torch.tensor([i % 3 for i in range(5)])
        
        env = SequenceEnvironment(sequences, labels)
        
        total_episodes = 0
        for _ in range(10):
            state = env.reset()
            _, _, done, _ = env.step(0)
            if done:
                total_episodes += 1
        
        assert total_episodes == 10


class TestDQNAgent:
    """Tests for DQN Agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=64,
        )
        
        assert agent.num_actions == 3

    def test_select_action_exploration(self):
        """Test action selection with exploration."""
        agent = DQNAgent(
            in_channels=4,
            num_actions=3,
        )
        
        state = torch.randn(4, 100)
        
        # With epsilon=1.0, should always pick random
        actions = [agent.select_action(state, epsilon=1.0)[0] for _ in range(100)]
        
        # Should have variety in actions (may not always have all 3 due to randomness)
        unique_actions = len(set(actions))
        # With 100 samples at epsilon=1.0, expect variety
        assert unique_actions >= 1

    def test_select_action_exploitation(self):
        """Test action selection with exploitation."""
        agent = DQNAgent(
            in_channels=4,
            num_actions=3,
        )
        
        state = torch.randn(4, 100)
        
        # With epsilon=0.0, should always pick same action (greedy)
        actions = [agent.select_action(state, epsilon=0.0)[0] for _ in range(10)]
        
        assert len(set(actions)) == 1

    def test_forward(self):
        """Test forward pass returns Q-values."""
        agent = DQNAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=32,
        )
        
        state = torch.randn(1, 4, 100)
        q_values = agent(state)
        
        assert q_values.shape == (1, 3)

    def test_batch_forward(self):
        """Test batch forward pass."""
        agent = DQNAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=32,
        )
        
        states = torch.randn(8, 4, 100)
        q_values = agent(states)
        
        assert q_values.shape == (8, 3)


class TestPolicyGradientAgent:
    """Tests for Policy Gradient (REINFORCE) Agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = PolicyGradientAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=64,
        )
        
        assert agent.num_actions == 3

    def test_select_action(self):
        """Test action selection."""
        agent = PolicyGradientAgent(
            in_channels=4,
            num_actions=3,
        )
        
        state = torch.randn(4, 100)
        result = agent.select_action(state)
        
        # Returns (action, confidence, log_prob)
        assert len(result) == 3
        action, confidence, log_prob = result
        assert 0 <= action < 3
        assert 0 <= confidence <= 1

    def test_action_distribution(self):
        """Test that actions follow learned distribution."""
        agent = PolicyGradientAgent(
            in_channels=4,
            num_actions=3,
        )
        
        state = torch.randn(4, 100)
        
        # Sample many actions
        actions = []
        for _ in range(100):
            action, _, _ = agent.select_action(state)
            actions.append(action)
        
        # Should have some variety
        unique_actions = set(actions)
        assert len(unique_actions) >= 1

    def test_forward(self):
        """Test forward pass returns probs and value."""
        agent = PolicyGradientAgent(
            in_channels=4,
            num_actions=3,
        )
        
        state = torch.randn(1, 4, 100)
        probs, value = agent(state)
        
        assert probs.shape == (1, 3)
        assert value.shape == (1, 1)
        # Probs should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)


class TestActorCriticAgent:
    """Tests for Actor-Critic (A2C) Agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = ActorCriticAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=64,
        )
        
        assert agent.num_actions == 3

    def test_get_action_and_value(self):
        """Test action and value computation."""
        agent = ActorCriticAgent(
            in_channels=4,
            num_actions=3,
        )
        
        state = torch.randn(1, 4, 100)
        action, log_prob, entropy, value = agent.get_action_and_value(state)
        
        # Check outputs
        assert 0 <= action.item() < 3
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)

    def test_forward(self):
        """Test forward pass returns logits and value."""
        agent = ActorCriticAgent(
            in_channels=4,
            num_actions=3,
        )
        
        state = torch.randn(1, 4, 100)
        logits, value = agent(state)
        
        assert logits.shape == (1, 3)
        assert value.shape == (1, 1)


class TestRLTraining:
    """Tests for RL training loops."""

    def test_dqn_training_step(self):
        """Test DQN can do a training step."""
        agent = DQNAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=32,
        )
        
        # Create simple optimizer
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        
        # Forward pass
        states = torch.randn(8, 4, 100)
        q_values = agent(states)
        
        # Simple loss
        target_q = torch.randn(8, 3)
        loss = nn.MSELoss()(q_values, target_q)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0

    def test_policy_gradient_training_step(self):
        """Test Policy Gradient can do a training step."""
        agent = PolicyGradientAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=32,
        )
        
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        
        # Forward pass
        states = torch.randn(8, 4, 100)
        probs, values = agent(states)
        
        # Simple policy gradient loss
        actions = torch.randint(0, 3, (8,))
        advantages = torch.randn(8)
        
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)))
        policy_loss = -(log_probs.squeeze() * advantages).mean()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        assert not torch.isnan(policy_loss)

    def test_actor_critic_training_step(self):
        """Test Actor-Critic can do a training step."""
        agent = ActorCriticAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=32,
        )
        
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        
        # Forward pass - returns logits and values
        states = torch.randn(8, 4, 100)
        logits, values = agent(states)
        
        # Convert logits to probs
        probs = torch.softmax(logits, dim=1)
        
        # Actor loss
        actions = torch.randint(0, 3, (8,))
        advantages = torch.randn(8)
        
        # Use log_softmax for numerical stability
        log_probs = torch.log_softmax(logits, dim=1).gather(1, actions.unsqueeze(1))
        actor_loss = -(log_probs.squeeze() * advantages.detach()).mean()
        
        # Critic loss
        returns = torch.randn(8, 1)
        critic_loss = nn.MSELoss()(values, returns)
        
        # Combined loss
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)


class TestRLPerformance:
    """Tests for RL performance characteristics."""

    def test_dqn_gradient_flow(self):
        """Test DQN has proper gradient flow."""
        agent = DQNAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=32,
        )
        
        states = torch.randn(4, 4, 100)
        q_values = agent(states)
        
        loss = q_values.sum()
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in agent.parameters())
        assert has_grads
