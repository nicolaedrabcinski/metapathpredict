"""
Integration tests for MetaPathPredict.

These tests verify that different components work together correctly.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from metapathpredict.config.settings import Settings, DataConfig, ModelConfig, TrainingConfig
from metapathpredict.data.preprocessing import OneHotEncoder, SequencePreprocessor, fragment_sequence
from metapathpredict.models.cnn import MultiScaleCNN
from metapathpredict.models.configurable_cnn import create_configurable_cnn, KERNEL_PRESETS
from metapathpredict.models.contrastive import ContrastiveEncoder, NTXentLoss, ContrastiveAugmentation
from metapathpredict.models.reinforcement import DQNAgent, SequenceEnvironment


@pytest.fixture
def sample_dna_sequences():
    """Generate realistic DNA sequences."""
    np.random.seed(42)
    bases = ["A", "C", "G", "T"]
    sequences = []
    labels = []
    
    for i in range(30):
        # Generate 1500 nt sequence (will be fragmented to 1000)
        seq = "".join(np.random.choice(bases, size=1500))
        sequences.append(seq)
        labels.append(i % 3)  # 0=virus, 1=bacteria, 2=eukaryotic
    
    return sequences, labels


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    return SequencePreprocessor(min_length=100)


@pytest.fixture
def encoder():
    """Create one-hot encoder."""
    return OneHotEncoder()


class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    def test_preprocessing_to_model_pipeline(self, sample_dna_sequences, preprocessor, encoder):
        """Test complete pipeline from raw sequences to model prediction."""
        sequences, labels = sample_dna_sequences
        
        # Step 1: Preprocess sequences
        processed_sequences = []
        processed_labels = []
        
        for seq, label in zip(sequences, labels):
            # process returns (cleaned_seq or None, stats)
            cleaned_seq, stats = preprocessor.process(seq)
            if stats.is_valid:
                # Fragment to fixed size
                fragments = fragment_sequence(
                    cleaned_seq,
                    fragment_size=1000,
                    step_size=1000
                )
                for frag in fragments:
                    processed_sequences.append(frag)
                    processed_labels.append(label)
        
        assert len(processed_sequences) > 0
        
        # Step 2: Encode sequences
        encoded = []
        for seq in processed_sequences[:16]:  # Use 16 for batch
            enc = encoder.encode(seq)
            encoded.append(enc)
        
        # Stack and transpose to (batch, channels, seq_len)
        batch = np.stack(encoded)
        batch = np.transpose(batch, (0, 2, 1))  # (B, 4, 1000)
        batch_tensor = torch.from_numpy(batch).float()
        
        assert batch_tensor.shape == (16, 4, 1000)
        
        # Step 3: Model prediction - MultiScaleCNN uses seq_length
        model = MultiScaleCNN(
            seq_length=1000,
            num_classes=3,
            branch_channels=32,
        )
        model.eval()
        
        with torch.no_grad():
            predictions = model(batch_tensor)
        
        assert predictions.shape == (16, 3)
        
        # Step 4: Get class predictions
        predicted_classes = torch.argmax(predictions, dim=1)
        assert predicted_classes.shape == (16,)
        assert all(0 <= c < 3 for c in predicted_classes)

class TestConfigurableCNNIntegration:
    """Integration tests for configurable CNN with different kernel sizes."""

    @pytest.mark.parametrize("preset", ["small", "medium", "large", "multi"])
    def test_all_presets_work_in_pipeline(self, preset, encoder):
        """Test all kernel presets work in complete pipeline."""
        # Generate data
        np.random.seed(42)
        sequences = ["".join(np.random.choice(list("ACGT"), size=1000)) for _ in range(8)]
        
        # Encode
        encoded = []
        for seq in sequences:
            enc = encoder.encode(seq)
            encoded.append(enc)
        
        batch = np.stack(encoded)
        batch = np.transpose(batch, (0, 2, 1))
        batch_tensor = torch.from_numpy(batch).float()
        
        # Create model with preset - use kernel_size parameter (string)
        model = create_configurable_cnn(
            kernel_size=preset,
            in_channels=4,
            num_classes=3,
        )
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch_tensor)
        
        assert output.shape == (8, 3)
        
        # Verify softmax produces valid probabilities
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


class TestContrastiveLearningIntegration:
    """Integration tests for contrastive learning pipeline."""

    def test_contrastive_pretraining_then_classification(self, encoder):
        """Test contrastive pretraining followed by classification."""
        np.random.seed(42)
        
        # Generate data
        sequences = ["".join(np.random.choice(list("ACGT"), size=500)) for _ in range(16)]
        labels = [i % 3 for i in range(16)]
        
        # Encode
        encoded = []
        for seq in sequences:
            enc = encoder.encode(seq)
            encoded.append(enc)
        
        batch = np.stack(encoded)
        batch = np.transpose(batch, (0, 2, 1))
        X = torch.from_numpy(batch).float()
        y = torch.tensor(labels)
        
        # Step 1: Contrastive pretraining - use correct parameters
        contrastive_encoder = ContrastiveEncoder(
            in_channels=4,
            backbone="medium",
            projection_dim=32,
        )
        
        augmentation = ContrastiveAugmentation()
        loss_fn = NTXentLoss(temperature=0.5)
        optimizer = torch.optim.Adam(contrastive_encoder.parameters(), lr=0.01)
        
        # Pretrain for a few steps
        contrastive_encoder.train()
        for _ in range(5):
            optimizer.zero_grad()
            
            # ContrastiveAugmentation returns (view1, view2)
            x_i, x_j = augmentation(X)
            
            z_i = contrastive_encoder(x_i)
            z_j = contrastive_encoder(x_j)
            
            loss = loss_fn(z_i, z_j)
            loss.backward()
            optimizer.step()
        
        # Step 2: Extract embeddings
        contrastive_encoder.eval()
        with torch.no_grad():
            embeddings = contrastive_encoder.get_embeddings(X)
        
        assert embeddings.shape[0] == 16
        
        # Step 3: Train classifier on embeddings
        embed_dim = embeddings.shape[1]
        classifier = torch.nn.Linear(embed_dim, 3)
        clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(10):
            clf_optimizer.zero_grad()
            logits = classifier(embeddings)
            loss = criterion(logits, y)
            loss.backward()
            clf_optimizer.step()
        
        # Verify classifier works
        with torch.no_grad():
            predictions = classifier(embeddings)
            predicted_classes = torch.argmax(predictions, dim=1)
        
        assert predicted_classes.shape == (16,)


class TestReinforcementLearningIntegration:
    """Integration tests for reinforcement learning pipeline."""

    def test_rl_training_loop(self, encoder):
        """Test complete RL training loop."""
        np.random.seed(42)
        
        # Generate data
        sequences = ["".join(np.random.choice(list("ACGT"), size=100)) for _ in range(20)]
        # Labels need to be tensors for SequenceEnvironment
        labels = torch.tensor([i % 3 for i in range(20)])
        
        # Encode sequences
        encoded_seqs = []
        for seq in sequences:
            enc = encoder.encode(seq)
            encoded_seqs.append(torch.from_numpy(enc.T).float())  # (4, 100)
        
        # Create environment - labels must be tensor
        env = SequenceEnvironment(encoded_seqs, labels)
        
        # Create agent - uses in_channels, num_actions
        agent = DQNAgent(
            in_channels=4,
            num_actions=3,
            hidden_dim=32,
        )
        
        # Simple training loop
        total_rewards = []
        
        for episode in range(10):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # select_action returns (action, confidence)
                action, _ = agent.select_action(state, epsilon=0.5)
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        # Verify training completed
        assert len(total_rewards) == 10


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_config_to_model_creation(self, tmp_path):
        """Test creating model from configuration."""
        # Create config with correct attribute names
        config = Settings(
            data=DataConfig(default_fragment_size=500, batch_size=16),
            model=ModelConfig(
                hidden_dims=[64, 32],
                num_classes=3,
                dropout_rate=0.3,
            ),
            training=TrainingConfig(
                num_epochs=10,
                learning_rate=0.001,
            ),
        )
        
        # Save and load config
        config_path = tmp_path / "config.yaml"
        config.to_yaml(str(config_path))
        loaded_config = Settings.from_yaml(str(config_path))
        
        # Create model from config
        model = MultiScaleCNN(
            seq_length=loaded_config.data.default_fragment_size,
            num_classes=loaded_config.model.num_classes,
            hidden_dims=loaded_config.model.hidden_dims,
            dropout=loaded_config.model.dropout_rate,
        )
        
        # Verify model matches config
        x = torch.randn(8, 4, loaded_config.data.default_fragment_size)
        output = model(x)
        
        assert output.shape == (8, loaded_config.model.num_classes)


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""

    def test_fasta_to_training_data(self, tmp_path, encoder):
        """Test converting FASTA to training-ready data."""
        # Create mock FASTA file
        fasta_content = """>seq1 virus
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
>seq2 bacteria
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA
>seq3 eukaryotic
AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCC
AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCC
AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCC
"""
        
        fasta_path = tmp_path / "test.fasta"
        fasta_path.write_text(fasta_content)
        
        # Parse FASTA (simplified)
        sequences = []
        labels = []
        current_seq = ""
        current_label = None
        
        label_map = {"virus": 0, "bacteria": 1, "eukaryotic": 2}
        
        for line in fasta_content.strip().split("\n"):
            if line.startswith(">"):
                if current_seq and current_label is not None:
                    sequences.append(current_seq)
                    labels.append(current_label)
                
                # Parse header
                for key in label_map:
                    if key in line.lower():
                        current_label = label_map[key]
                        break
                current_seq = ""
            else:
                current_seq += line.strip()
        
        # Don't forget last sequence
        if current_seq and current_label is not None:
            sequences.append(current_seq)
            labels.append(current_label)
        
        assert len(sequences) == 3
        assert labels == [0, 1, 2]
        
        # Encode and create batch
        preprocessor = SequencePreprocessor(min_length=50)
        
        encoded_data = []
        for seq in sequences:
            # process returns (cleaned_seq or None, stats)
            cleaned_seq, stats = preprocessor.process(seq)
            if stats.is_valid:
                enc = encoder.encode(cleaned_seq)
                encoded_data.append(enc)
        
        assert len(encoded_data) == 3


class TestMultiModelEnsemble:
    """Integration tests for ensemble predictions."""

    def test_ensemble_of_different_kernels(self, encoder):
        """Test ensemble of models with different kernel sizes."""
        np.random.seed(42)
        
        # Generate test data
        sequences = ["".join(np.random.choice(list("ACGT"), size=500)) for _ in range(8)]
        
        encoded = []
        for seq in sequences:
            enc = encoder.encode(seq)
            encoded.append(enc)
        
        batch = np.stack(encoded)
        batch = np.transpose(batch, (0, 2, 1))
        X = torch.from_numpy(batch).float()
        
        # Create models with different kernels - use kernel_size parameter
        models = []
        for preset in ["small", "medium", "large"]:
            model = create_configurable_cnn(
                kernel_size=preset,
                in_channels=4,
                num_classes=3,
            )
            model.eval()
            models.append(model)
        
        # Get predictions from each model
        all_predictions = []
        with torch.no_grad():
            for model in models:
                preds = model(X)
                probs = torch.softmax(preds, dim=1)
                all_predictions.append(probs)
        
        # Ensemble by averaging
        stacked = torch.stack(all_predictions)
        ensemble_probs = stacked.mean(dim=0)
        
        assert ensemble_probs.shape == (8, 3)
        
        # Verify probabilities sum to 1
        assert torch.allclose(ensemble_probs.sum(dim=1), torch.ones(8), atol=1e-5)
        
        # Get final predictions
        final_predictions = torch.argmax(ensemble_probs, dim=1)
        assert final_predictions.shape == (8,)
