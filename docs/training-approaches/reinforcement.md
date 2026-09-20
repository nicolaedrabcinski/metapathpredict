# Reinforcement Learning

**Read this before using it:** the task here is a one-step contextual bandit — one fragment, one
action (predict a class), one reward, no next state — not a multi-step sequential-decision problem.
Most of what makes RL different from a classifier (exploration/exploitation over time, credit
assignment across steps, planning) has nothing to act on in this setup. In every run so far, RL
fine-tuning from a contrastive checkpoint has not beaten the contrastive linear probe it started
from, and neither has beaten a plain CNN trained directly on the labels — see
`benchmarks/README.md` for current numbers. Treat this as an experiment, not the default choice.

## Agents

```python
from metapathpredict.models.reinforcement import (
    DQNAgent, PolicyGradientAgent, ActorCriticAgent, SequenceEnvironment, RLTrainer,
)

agent = ActorCriticAgent(
    num_actions=8, backbone="large", base_channels=128, hidden_dim=256, norm="batch",
)
```

All three agents (`DQNAgent`, `PolicyGradientAgent`, `ActorCriticAgent`) wrap the same
`ConfigurableCNN` encoder as the contrastive/plain-CNN paths (`backbone`/`base_channels`/`norm` mean
the same thing), with a policy (and, for DQN/actor-critic, a value) head on top. To continue from a
contrastive checkpoint, `base_channels`/`backbone`/`norm` must match the checkpoint's — the CLI's
`train --pipeline full` does this for you.

## Environment and reward

```python
env = SequenceEnvironment(
    sequences=train_dataset,   # a Dataset (lazy) or a [N, 4, length] tensor
    reward_correct=1.0, reward_incorrect=-0.5, reward_uncertain=-0.1,
)
```

One "episode" is one fragment: `env.reset()` returns its encoding, the agent picks a class, the
episode ends immediately with the reward. `reward_uncertain` only applies to agents that can abstain
(not all of them do); most runs just use correct/incorrect.

## Training

```bash
metapathpredict train --pipeline rl --config configs/train_gpu.yaml    # needs an existing encoder checkpoint
metapathpredict train --pipeline full --config configs/train_gpu.yaml  # contrastive pretrain, then RL
```

reads the `rl:` block of the config (`algorithm`, `backbone`, `hidden_dim`, the three rewards,
`num_epochs`, `episodes_per_epoch`, `learning_rate`, `gamma`, `early_stopping_patience`, and for DQN
the epsilon-greedy schedule / replay buffer size). Programmatically:

```python
trainer = RLTrainer(agent, env, optimizer, device="cuda", gamma=0.99, algorithm="actor_critic")
metrics = trainer.train_epoch(num_episodes=64000, batch_size=32)  # avg_reward, accuracy, avg_loss
```

For `algorithm="actor_critic"` with `batch_size > 1`, `train_epoch` batches transitions
(`train_batch_actor_critic`) instead of updating one episode at a time — this is what the current
configs use (`episodes_per_epoch: 64000`, `rl.batch_size: 32`), since with a one-step bandit there is
no trajectory to keep sequential.
