# API Reference: Models

Neural network models for sequence classification.

## ConfigurableCNN

```python
class ConfigurableCNN(nn.Module):
    """Configurable CNN for DNA sequence classification.
    
    Supports three kernel size presets to capture patterns at different scales.
    
    Args:
        input_channels: Number of input channels (4 for one-hot).
        num_classes: Number of output classes.
        kernel_preset: Kernel size preset ("small"=5, "medium"=7, "large"=10).
        hidden_channels: List of hidden channel dimensions.
        dropout: Dropout rate.
    
    Attributes:
        KERNEL_PRESETS: Mapping of preset names to kernel sizes.
    
    Example:
        >>> model = ConfigurableCNN(kernel_preset="medium", num_classes=3)
        >>> x = torch.randn(8, 4, 500)
        >>> output = model(x)
        >>> output.shape
        torch.Size([8, 3])
    """
    
    KERNEL_PRESETS = {"small": 5, "medium": 7, "large": 10}
    
    def __init__(
        self,
        input_channels: int = 4,
        num_classes: int = 3,
        kernel_preset: str = "medium",
        hidden_channels: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, length).
        
        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        pass
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer.
        
        Args:
            x: Input tensor.
        
        Returns:
            Feature tensor of shape (batch, hidden_channels[-1]).
        """
        pass
```

## MultiScaleCNN

```python
class MultiScaleCNN(nn.Module):
    """Multi-scale CNN with parallel kernel branches.
    
    Combines multiple kernel sizes and concatenates features.
    
    Args:
        kernel_sizes: List of kernel sizes to use.
        hidden_channels: Hidden channel dimensions per branch.
        num_classes: Number of output classes.
    
    Example:
        >>> model = MultiScaleCNN(kernel_sizes=[5, 7, 10], num_classes=3)
        >>> output = model(x)
    """
    
    def __init__(
        self,
        kernel_sizes: List[int] = [5, 7, 10],
        hidden_channels: List[int] = [32, 64, 128],
        num_classes: int = 3,
    ):
        pass
```

## ContrastiveModel

```python
class ContrastiveModel(nn.Module):
    """Contrastive learning model with projection head.
    
    Implements SimCLR-style contrastive learning.
    
    Args:
        encoder: Backbone encoder network.
        projection_dim: Output dimension of projection head.
        hidden_dim: Hidden dimension in projection head.
    
    Example:
        >>> encoder = ConfigurableCNN(kernel_preset="large")
        >>> model = ContrastiveModel(encoder, projection_dim=128)
        >>> z = model(x)  # Projected embeddings
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 256,
    ):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to contrastive space.
        
        Returns:
            Normalized embeddings of shape (batch, projection_dim).
        """
        pass
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder features (before projection)."""
        pass
```

## ProjectionHead

```python
class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.
    
    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output embedding dimension.
    
    Example:
        >>> proj = ProjectionHead(256, 256, 128)
        >>> z = proj(features)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        pass
```

## Loss Functions

### SimCLRLoss

```python
class SimCLRLoss(nn.Module):
    """NT-Xent loss for SimCLR.
    
    Args:
        temperature: Temperature scaling parameter.
    
    Example:
        >>> criterion = SimCLRLoss(temperature=0.5)
        >>> loss = criterion(z1, z2)
    """
    
    def __init__(self, temperature: float = 0.5):
        pass
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.
        
        Args:
            z1: Embeddings from first augmented view.
            z2: Embeddings from second augmented view.
        
        Returns:
            Scalar loss value.
        """
        pass
```

### SupConLoss

```python
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.
    
    Uses labels to pull same-class samples together.
    
    Args:
        temperature: Temperature parameter.
    
    Example:
        >>> criterion = SupConLoss(temperature=0.07)
        >>> loss = criterion(embeddings, labels)
    """
    
    def __init__(self, temperature: float = 0.07):
        pass
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        pass
```

## Reinforcement Learning

### PolicyGradientAgent

```python
class PolicyGradientAgent(nn.Module):
    """REINFORCE policy gradient agent.
    
    Args:
        state_dim: State/observation dimension.
        action_dim: Number of actions (classes).
        hidden_dim: Hidden layer dimension.
    
    Example:
        >>> agent = PolicyGradientAgent(256, 3, 128)
        >>> action, log_prob = agent.select_action(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        pass
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        pass
    
    def select_action(
        self,
        state: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """Sample action from policy.
        
        Returns:
            Tuple of (action, log_probability).
        """
        pass
```

### DQNAgent

```python
class DQNAgent(nn.Module):
    """Deep Q-Network agent.
    
    Args:
        state_dim: State dimension.
        action_dim: Number of actions.
        hidden_dim: Hidden layer dimension.
        epsilon: Exploration rate.
        gamma: Discount factor.
    
    Example:
        >>> agent = DQNAgent(256, 3, epsilon=0.1)
        >>> action = agent.select_action(state, epsilon=0.1)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        epsilon: float = 0.1,
        gamma: float = 0.99,
    ):
        pass
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions."""
        pass
    
    def select_action(
        self,
        state: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> int:
        """Select action using epsilon-greedy."""
        pass
    
    def update(
        self,
        batch: Tuple,
        target_network: "DQNAgent",
    ) -> float:
        """Update Q-network from batch."""
        pass
```

### ActorCriticAgent

```python
class ActorCriticAgent(nn.Module):
    """Actor-Critic (A2C) agent.
    
    Args:
        state_dim: State dimension.
        action_dim: Number of actions.
        hidden_dim: Hidden layer dimension.
    
    Example:
        >>> agent = ActorCriticAgent(256, 3)
        >>> action, log_prob, value = agent.select_action(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        pass
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action probabilities and state value."""
        pass
    
    def select_action(
        self,
        state: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action and return (action, log_prob, value)."""
        pass
```

### SequenceClassificationEnv

```python
class SequenceClassificationEnv:
    """RL environment for sequence classification.
    
    Args:
        sequences: Encoded sequences tensor.
        labels: Class labels tensor.
        encoder: Feature encoder network.
        reward_fn: Optional custom reward function.
    
    Example:
        >>> env = SequenceClassificationEnv(X, y, encoder)
        >>> state = env.reset()
        >>> next_state, reward, done, info = env.step(action)
    """
    
    def __init__(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor,
        encoder: nn.Module,
        reward_fn: Optional[Callable] = None,
    ):
        pass
    
    def reset(self) -> torch.Tensor:
        """Reset to random sequence, return initial state."""
        pass
    
    def step(
        self,
        action: int,
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take action, return (next_state, reward, done, info)."""
        pass
```

## Utility Classes

### ReplayBuffer

```python
class ReplayBuffer:
    """Experience replay buffer for DQN.
    
    Args:
        capacity: Maximum buffer size.
    """
    
    def __init__(self, capacity: int = 10000):
        pass
    
    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store transition."""
        pass
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch."""
        pass
    
    def __len__(self) -> int:
        """Return current buffer size."""
        pass
```
