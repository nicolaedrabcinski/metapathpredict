"""
Real ML Model Interpretation Service.

Implements state-of-the-art interpretability methods:
- Integrated Gradients for nucleotide-level attribution
- Attention weight extraction and analysis
- Grad-CAM for CNN layer visualization  
- Layer-wise Relevance Propagation (LRP)
- SHAP-style feature importance
- CNN filter motif extraction
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Interpretation Results
# ============================================================================

@dataclass
class IntegratedGradientsResult:
    """Result of Integrated Gradients attribution."""
    attributions: np.ndarray  # (seq_len, 4) - per nucleotide importance
    convergence_delta: float
    baseline_prediction: np.ndarray
    input_prediction: np.ndarray
    
    @property
    def position_importance(self) -> np.ndarray:
        """Sum attributions across nucleotide channels."""
        return np.sum(np.abs(self.attributions), axis=-1)
    
    def top_positions(self, k: int = 20) -> list[tuple[int, float]]:
        """Get top-k most important positions."""
        importance = self.position_importance
        indices = np.argsort(importance)[::-1][:k]
        return [(int(idx), float(importance[idx])) for idx in indices]


@dataclass  
class AttentionAnalysis:
    """Analysis of attention patterns."""
    layer_attention_weights: list[np.ndarray]  # Per-layer attention
    head_attention_weights: list[np.ndarray]  # Per-head attention
    position_importance: np.ndarray  # Aggregated position importance
    attention_entropy: list[float]  # Per-layer entropy
    attention_sparsity: list[float]  # Per-layer sparsity
    
    @property
    def num_layers(self) -> int:
        return len(self.layer_attention_weights)
    
    def get_head_specialization(self, layer: int) -> dict:
        """Analyze what each head focuses on."""
        if layer >= len(self.head_attention_weights):
            return {}
        
        heads = self.head_attention_weights[layer]
        specialization = {}
        
        for h, head_attn in enumerate(heads):
            # Compute attention statistics
            mean_pos = np.average(np.arange(len(head_attn)), weights=head_attn)
            std_pos = np.sqrt(np.average((np.arange(len(head_attn)) - mean_pos)**2, weights=head_attn))
            max_pos = int(np.argmax(head_attn))
            entropy = -np.sum(head_attn * np.log(head_attn + 1e-10))
            
            specialization[f"head_{h}"] = {
                "focus_position": float(mean_pos),
                "focus_spread": float(std_pos),
                "peak_position": max_pos,
                "entropy": float(entropy),
                "is_local": std_pos < len(head_attn) * 0.1,
            }
        
        return specialization


@dataclass
class GradCAMResult:
    """Result of Grad-CAM analysis."""
    heatmaps: dict[str, np.ndarray]  # Layer name -> heatmap
    class_activations: dict[str, float]
    important_regions: list[tuple[int, int, float]]  # (start, end, importance)


@dataclass
class MotifInfo:
    """Information about discovered motif."""
    pattern: str
    consensus: str  
    pwm: np.ndarray  # Position weight matrix
    information_content: float
    occurrences: list[int]
    avg_importance: float
    class_specificity: dict[str, float]


@dataclass
class CNNFilterAnalysis:
    """Analysis of CNN filter activations."""
    filter_motifs: list[MotifInfo]
    filter_activations: np.ndarray  # (num_filters, seq_len)
    top_activating_regions: list[tuple[int, int, int, float]]  # (filter_id, start, end, activation)


# ============================================================================
# Model Interpreter Core
# ============================================================================

class ModelInterpreter:
    """
    Comprehensive model interpretation for sequence classification.
    
    Supports multiple interpretation methods:
    - integrated_gradients: Axiomatic attribution method
    - attention: Extract and analyze attention patterns
    - gradcam: CNN activation visualization
    - filter_motifs: Extract motifs from CNN filters
    - lrp: Layer-wise relevance propagation
    """
    
    NUCLEOTIDE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    NUCLEOTIDES = ['A', 'C', 'G', 'T']
    CLASS_NAMES = ['bacteria', 'eukaryotic', 'virus']
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        seq_length: int = 1000,
    ):
        """
        Initialize interpreter.
        
        Args:
            model: The trained classification model
            device: Device to run computations on
            seq_length: Expected sequence length
        """
        self.model = model
        self.device = torch.device(device)
        self.seq_length = seq_length
        self.model.to(self.device)
        self.model.eval()
        
        # Hook storage for attention/activation extraction
        self._attention_weights: list[Tensor] = []
        self._activations: dict[str, Tensor] = {}
        self._gradients: dict[str, Tensor] = {}
        self._hooks: list = []
    
    def _encode_sequence(self, sequence: str) -> Tensor:
        """One-hot encode DNA sequence."""
        seq = sequence.upper()[:self.seq_length]
        
        # Pad if necessary
        if len(seq) < self.seq_length:
            seq = seq + 'N' * (self.seq_length - len(seq))
        
        # One-hot encode
        encoding = np.zeros((len(seq), 4), dtype=np.float32)
        for i, nuc in enumerate(seq):
            if nuc in self.NUCLEOTIDE_MAP:
                encoding[i, self.NUCLEOTIDE_MAP[nuc]] = 1.0
            else:
                encoding[i, 0] = 0.25  # Unknown -> uniform
                encoding[i, 1] = 0.25
                encoding[i, 2] = 0.25
                encoding[i, 3] = 0.25
        
        return torch.tensor(encoding, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _decode_position(self, encoding: np.ndarray) -> str:
        """Decode one-hot position to nucleotide."""
        idx = np.argmax(encoding)
        return self.NUCLEOTIDES[idx]
    
    # ========================================================================
    # Integrated Gradients
    # ========================================================================
    
    def integrated_gradients(
        self,
        sequence: str,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        baseline: Optional[str] = None,
    ) -> IntegratedGradientsResult:
        """
        Compute Integrated Gradients attribution.
        
        Integrated Gradients satisfies key axioms:
        - Sensitivity: If input differs from baseline at one feature, 
          and model output changes, that feature gets non-zero attribution
        - Implementation Invariance: Two networks computing same function 
          get same attributions
        
        Args:
            sequence: Input DNA sequence
            target_class: Class to explain (None = predicted class)
            n_steps: Number of interpolation steps
            baseline: Baseline sequence (None = all zeros)
            
        Returns:
            IntegratedGradientsResult with attributions
        """
        # Encode input
        input_tensor = self._encode_sequence(sequence)
        input_tensor.requires_grad_(True)
        
        # Create baseline (zeros = no information)
        if baseline is None:
            baseline_tensor = torch.zeros_like(input_tensor)
        else:
            baseline_tensor = self._encode_sequence(baseline)
        
        # Get predictions for baseline and input
        with torch.no_grad():
            baseline_output = F.softmax(self.model(baseline_tensor), dim=-1)
            input_output = F.softmax(self.model(input_tensor), dim=-1)
        
        baseline_pred = baseline_output.cpu().numpy()[0]
        input_pred = input_output.cpu().numpy()[0]
        
        # Determine target class
        if target_class is None:
            target_class = int(torch.argmax(input_output, dim=-1).item())
        
        # Compute gradients along interpolation path
        scaled_inputs = []
        for step in range(n_steps + 1):
            alpha = step / n_steps
            scaled_input = baseline_tensor + alpha * (input_tensor - baseline_tensor)
            scaled_inputs.append(scaled_input)
        
        # Stack and compute gradients
        # detach: built from input_tensor (which requires grad), so the concatenation would be
        # a non-leaf and .grad below would always be None
        scaled_inputs = torch.cat(scaled_inputs, dim=0).detach()
        scaled_inputs.requires_grad_(True)
        
        outputs = self.model(scaled_inputs)
        
        # Get gradient w.r.t. target class
        target_scores = outputs[:, target_class]
        target_scores.sum().backward()
        
        gradients = scaled_inputs.grad  # (n_steps+1, seq_len, 4)
        
        # Approximate integral using trapezoidal rule
        avg_gradients = (gradients[:-1] + gradients[1:]).mean(dim=0) / 2
        
        # Integrated gradients = (input - baseline) * avg_gradients
        attributions = (input_tensor - baseline_tensor).detach() * avg_gradients
        attributions = attributions.squeeze(0).cpu().numpy()
        
        # Compute convergence delta (should be close to 0)
        attr_sum = np.sum(attributions)
        pred_diff = input_pred[target_class] - baseline_pred[target_class]
        convergence_delta = abs(attr_sum - pred_diff)
        
        return IntegratedGradientsResult(
            attributions=attributions,
            convergence_delta=convergence_delta,
            baseline_prediction=baseline_pred,
            input_prediction=input_pred,
        )
    
    # ========================================================================
    # Attention Analysis
    # ========================================================================
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        self._attention_weights = []
        
        def attention_hook(module, input, output):
            # Capture attention weights from SelfAttention
            if hasattr(module, 'attn'):
                # Store the attention pattern
                self._attention_weights.append(output.detach().cpu())
        
        # Find and hook attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                self._hooks.append(hook)
    
    def _extract_attention_from_model(self, input_tensor: Tensor) -> list[np.ndarray]:
        """Extract attention weights using hooks or model internals."""
        attention_weights = []
        
        # Try to access attention blocks directly
        if hasattr(self.model, 'attention_blocks'):
            # Store intermediate representations
            x = input_tensor
            
            # Run through stem and branches
            if x.dim() == 3 and x.shape[-1] == 4:
                x = x.transpose(1, 2)
            
            x = self.model.stem(x)
            branch_outputs = [branch(x) for branch in self.model.branches]
            x = torch.cat(branch_outputs, dim=1)
            x = self.model.fusion(x)
            x = self.model.fusion_se(x)
            
            # Transpose for attention
            x = x.transpose(1, 2)
            x = self.model.pos_encoding(x)
            
            # Process through attention blocks with weight extraction
            for attn_block in self.model.attention_blocks:
                # Get attention weights from self-attention
                attn_module = attn_block.attn
                
                B, N, C = x.shape
                qkv = attn_module.qkv(attn_block.norm1(x))
                qkv = qkv.reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                
                # Compute attention weights
                attn = (q @ k.transpose(-2, -1)) * attn_module.scale
                attn = F.softmax(attn, dim=-1)
                
                attention_weights.append(attn.detach().cpu().numpy())
                
                # Continue forward pass
                x = attn_block(x)
        
        return attention_weights
    
    def analyze_attention(
        self,
        sequence: str,
        aggregate_heads: bool = True,
    ) -> AttentionAnalysis:
        """
        Extract and analyze attention patterns.
        
        Args:
            sequence: Input DNA sequence
            aggregate_heads: Whether to aggregate across heads
            
        Returns:
            AttentionAnalysis with attention statistics
        """
        input_tensor = self._encode_sequence(sequence)
        
        with torch.no_grad():
            attention_weights = self._extract_attention_from_model(input_tensor)
        
        if not attention_weights:
            # Return empty analysis if no attention found
            return AttentionAnalysis(
                layer_attention_weights=[],
                head_attention_weights=[],
                position_importance=np.zeros(len(sequence)),
                attention_entropy=[],
                attention_sparsity=[],
            )
        
        layer_attention = []
        head_attention = []
        entropies = []
        sparsities = []
        
        for layer_idx, attn in enumerate(attention_weights):
            # attn shape: (batch, num_heads, seq_len, seq_len)
            attn = attn[0]  # Remove batch dimension
            
            # Store per-head attention (average over positions)
            head_attn = attn.mean(axis=-2)  # (num_heads, seq_len)
            head_attention.append(head_attn)
            
            # Aggregate across heads
            if aggregate_heads:
                layer_attn = attn.mean(axis=0)  # (seq_len, seq_len)
            else:
                layer_attn = attn
            layer_attention.append(layer_attn)
            
            # Compute entropy (measure of attention spread)
            flat_attn = attn.reshape(-1)
            entropy = -np.sum(flat_attn * np.log(flat_attn + 1e-10))
            entropies.append(float(entropy))
            
            # Compute sparsity (fraction of near-zero attention)
            sparsity = float(np.mean(flat_attn < 0.01))
            sparsities.append(sparsity)
        
        # Aggregate position importance across all layers
        all_attn = np.stack([a.mean(axis=0) if a.ndim == 2 else a.mean(axis=(0, 1)) 
                            for a in layer_attention], axis=0)
        position_importance = all_attn.mean(axis=0).sum(axis=0)  # Sum received attention
        position_importance = position_importance[:len(sequence)]
        
        return AttentionAnalysis(
            layer_attention_weights=layer_attention,
            head_attention_weights=head_attention,
            position_importance=position_importance,
            attention_entropy=entropies,
            attention_sparsity=sparsities,
        )
    
    # ========================================================================
    # Grad-CAM
    # ========================================================================
    
    def _register_gradcam_hooks(self, layer_names: list[str]):
        """Register hooks for Grad-CAM."""
        self._activations = {}
        self._gradients = {}
        
        def save_activation(name):
            def hook(module, input, output):
                self._activations[name] = output.detach()
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                self._gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if any(ln in name for ln in layer_names):
                self._hooks.append(module.register_forward_hook(save_activation(name)))
                self._hooks.append(module.register_full_backward_hook(save_gradient(name)))
    
    def gradcam(
        self,
        sequence: str,
        target_class: Optional[int] = None,
        layer_names: Optional[list[str]] = None,
    ) -> GradCAMResult:
        """
        Compute Grad-CAM heatmaps.
        
        Grad-CAM uses gradients flowing into the target layer
        to produce a coarse localization map highlighting 
        important regions for prediction.
        
        Args:
            sequence: Input DNA sequence
            target_class: Class to explain
            layer_names: Layers to visualize
            
        Returns:
            GradCAMResult with heatmaps
        """
        if layer_names is None:
            layer_names = ['branches', 'fusion', 'stem']
        
        # Clear previous hooks
        self._clear_hooks()
        self._register_gradcam_hooks(layer_names)
        
        input_tensor = self._encode_sequence(sequence)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = int(torch.argmax(output, dim=-1).item())
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Compute Grad-CAM for each layer
        heatmaps = {}
        
        for name, activation in self._activations.items():
            if name not in self._gradients:
                continue
            
            gradient = self._gradients[name]
            
            # Global average pool gradients
            weights = gradient.mean(dim=-1, keepdim=True)  # (batch, channels, 1)
            
            # Weighted combination of activation maps
            cam = (weights * activation).sum(dim=1)  # (batch, seq_len)
            cam = F.relu(cam)  # ReLU to keep only positive
            
            # Normalize
            cam = cam[0].cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to sequence length if needed
            if len(cam) != len(sequence):
                cam = np.interp(
                    np.linspace(0, 1, len(sequence)),
                    np.linspace(0, 1, len(cam)),
                    cam
                )
            
            heatmaps[name] = cam
        
        # Get class activations
        class_activations = {
            self.CLASS_NAMES[i]: float(output[0, i].item())
            for i in range(output.shape[-1])
        }
        
        # Find important regions
        important_regions = self._find_important_regions(heatmaps, threshold=0.5)
        
        self._clear_hooks()
        
        return GradCAMResult(
            heatmaps=heatmaps,
            class_activations=class_activations,
            important_regions=important_regions,
        )
    
    def _find_important_regions(
        self,
        heatmaps: dict[str, np.ndarray],
        threshold: float = 0.5,
        min_length: int = 10,
    ) -> list[tuple[int, int, float]]:
        """Find contiguous important regions from heatmaps."""
        regions = []
        
        # Average heatmaps
        avg_heatmap = np.mean(list(heatmaps.values()), axis=0)
        
        # Find regions above threshold
        above = avg_heatmap > threshold
        
        in_region = False
        start = 0
        
        for i, is_above in enumerate(above):
            if is_above and not in_region:
                in_region = True
                start = i
            elif not is_above and in_region:
                in_region = False
                if i - start >= min_length:
                    importance = float(avg_heatmap[start:i].mean())
                    regions.append((start, i, importance))
        
        # Handle region at end
        if in_region and len(above) - start >= min_length:
            importance = float(avg_heatmap[start:].mean())
            regions.append((start, len(above), importance))
        
        # Sort by importance
        regions.sort(key=lambda x: x[2], reverse=True)
        
        return regions
    
    # ========================================================================
    # CNN Filter Motif Extraction
    # ========================================================================
    
    def extract_filter_motifs(
        self,
        sequences: list[str],
        layer_name: str = "stem",
        top_k: int = 10,
        activation_threshold: float = 0.5,
    ) -> CNNFilterAnalysis:
        """
        Extract motifs from CNN filter activations.
        
        For each filter, finds sequence regions that maximally
        activate the filter and constructs a PWM (Position Weight Matrix).
        
        Args:
            sequences: List of sequences to analyze
            layer_name: CNN layer to analyze
            top_k: Number of top motifs to return
            activation_threshold: Threshold for considering activation significant
            
        Returns:
            CNNFilterAnalysis with discovered motifs
        """
        self._clear_hooks()
        
        # Hook to capture activations
        activations_list = []
        
        def capture_hook(module, input, output):
            activations_list.append(output.detach().cpu())
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if layer_name in name and isinstance(module, (nn.Conv1d, nn.Sequential)):
                target_module = module
                break
        
        if target_module is None:
            logger.warning(f"Layer {layer_name} not found")
            return CNNFilterAnalysis(
                filter_motifs=[],
                filter_activations=np.array([]),
                top_activating_regions=[],
            )
        
        hook = target_module.register_forward_hook(capture_hook)
        
        # Process sequences
        for seq in sequences:
            input_tensor = self._encode_sequence(seq)
            with torch.no_grad():
                self.model(input_tensor)
        
        hook.remove()
        
        if not activations_list:
            return CNNFilterAnalysis(
                filter_motifs=[],
                filter_activations=np.array([]),
                top_activating_regions=[],
            )
        
        # Stack activations: (num_sequences, num_filters, seq_len)
        all_activations = torch.cat(activations_list, dim=0).numpy()
        
        # For each filter, find top-activating regions
        num_filters = all_activations.shape[1]
        filter_motifs = []
        top_regions = []
        
        for filter_idx in range(min(num_filters, top_k * 2)):
            filter_acts = all_activations[:, filter_idx, :]  # (num_seq, seq_len)
            
            # Find top activations
            max_acts = filter_acts.max(axis=1)
            top_seq_idx = np.argsort(max_acts)[::-1][:5]
            
            # Collect activating regions
            motif_sequences = []
            positions = []
            
            for seq_idx in top_seq_idx:
                act = filter_acts[seq_idx]
                max_pos = int(np.argmax(act))
                max_val = float(act[max_pos])
                
                if max_val < activation_threshold * act.max():
                    continue
                
                # Extract region around max activation
                # Kernel size estimation (usually 5-15 for DNA)
                kernel_size = 9
                start = max(0, max_pos - kernel_size // 2)
                end = min(len(sequences[seq_idx]), max_pos + kernel_size // 2 + 1)
                
                region_seq = sequences[seq_idx][start:end]
                if len(region_seq) >= 5:
                    motif_sequences.append(region_seq)
                    positions.append(max_pos)
                
                top_regions.append((filter_idx, start, end, max_val))
            
            if len(motif_sequences) >= 2:
                # Build PWM from aligned sequences
                motif = self._build_motif(motif_sequences, filter_idx, positions)
                if motif:
                    filter_motifs.append(motif)
        
        # Sort motifs by importance
        filter_motifs.sort(key=lambda m: m.avg_importance, reverse=True)
        
        return CNNFilterAnalysis(
            filter_motifs=filter_motifs[:top_k],
            filter_activations=all_activations.mean(axis=0),  # Average across sequences
            top_activating_regions=sorted(top_regions, key=lambda x: x[3], reverse=True)[:50],
        )
    
    def _build_motif(
        self,
        sequences: list[str],
        filter_idx: int,
        positions: list[int],
    ) -> Optional[MotifInfo]:
        """Build motif from aligned sequences."""
        if not sequences:
            return None
        
        # Align sequences (simple: use shortest length)
        min_len = min(len(s) for s in sequences)
        aligned = [s[:min_len] for s in sequences]
        
        # Build PWM
        pwm = np.zeros((min_len, 4))
        
        for seq in aligned:
            for i, nuc in enumerate(seq.upper()):
                if nuc in self.NUCLEOTIDE_MAP:
                    pwm[i, self.NUCLEOTIDE_MAP[nuc]] += 1
        
        # Normalize
        pwm = pwm / (len(aligned) + 1e-10)
        
        # Compute consensus
        consensus = ""
        for i in range(min_len):
            max_idx = int(np.argmax(pwm[i]))
            if pwm[i, max_idx] > 0.5:
                consensus += self.NUCLEOTIDES[max_idx]
            else:
                consensus += "N"
        
        # Compute information content
        background = np.array([0.25, 0.25, 0.25, 0.25])
        ic = 0
        for i in range(min_len):
            for j in range(4):
                if pwm[i, j] > 0:
                    ic += pwm[i, j] * np.log2(pwm[i, j] / background[j])
        
        return MotifInfo(
            pattern=aligned[0] if aligned else "",
            consensus=consensus,
            pwm=pwm,
            information_content=float(ic),
            occurrences=positions,
            avg_importance=float(ic / min_len) if min_len > 0 else 0,
            class_specificity={},  # Can be computed separately
        )
    
    # ========================================================================
    # Saliency Maps (Simple Gradient)
    # ========================================================================
    
    def saliency_map(
        self,
        sequence: str,
        target_class: Optional[int] = None,
        smooth_samples: int = 0,
        noise_std: float = 0.1,
    ) -> np.ndarray:
        """
        Compute vanilla gradient saliency map.
        
        Optionally uses SmoothGrad for noise reduction.
        
        Args:
            sequence: Input sequence
            target_class: Class to explain
            smooth_samples: Number of samples for SmoothGrad (0 = vanilla)
            noise_std: Noise standard deviation for SmoothGrad
            
        Returns:
            Saliency scores per position
        """
        input_tensor = self._encode_sequence(sequence)
        
        if smooth_samples > 0:
            # SmoothGrad
            all_grads = []
            for _ in range(smooth_samples):
                noisy_input = input_tensor + torch.randn_like(input_tensor) * noise_std
                noisy_input.requires_grad_(True)
                
                output = self.model(noisy_input)
                if target_class is None:
                    target_class = int(torch.argmax(output, dim=-1).item())
                
                self.model.zero_grad()
                output[0, target_class].backward()
                
                all_grads.append(noisy_input.grad.detach().cpu().numpy())
            
            gradients = np.mean(all_grads, axis=0)
        else:
            # Vanilla gradient
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = int(torch.argmax(output, dim=-1).item())
            
            self.model.zero_grad()
            output[0, target_class].backward()
            
            gradients = input_tensor.grad.detach().cpu().numpy()
        
        # Compute saliency as absolute gradient magnitude
        saliency = np.abs(gradients).sum(axis=-1).squeeze()
        
        return saliency[:len(sequence)]
    
    # ========================================================================
    # Layer-wise Relevance Propagation (LRP)
    # ========================================================================
    
    def lrp(
        self,
        sequence: str,
        target_class: Optional[int] = None,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Layer-wise Relevance Propagation.
        
        Distributes the prediction score back through the network
        to determine input relevance.
        
        Args:
            sequence: Input sequence
            target_class: Class to explain  
            epsilon: Stabilization constant
            
        Returns:
            Relevance scores per position
        """
        input_tensor = self._encode_sequence(sequence)
        
        # For now, use gradient * input as LRP approximation
        # Full LRP requires layer-specific rules
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(output, dim=-1).item())
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Gradient * Input (DeepLIFT-style approximation)
        relevance = (input_tensor.grad * input_tensor).detach().cpu().numpy()
        relevance = relevance.sum(axis=-1).squeeze()
        
        return relevance[:len(sequence)]
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_weights = []
        self._activations = {}
        self._gradients = {}
    
    def predict_with_confidence(
        self,
        sequence: str,
    ) -> tuple[int, np.ndarray, float]:
        """
        Get prediction with confidence analysis.
        
        Returns:
            (predicted_class, probabilities, entropy)
        """
        input_tensor = self._encode_sequence(sequence)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=-1).cpu().numpy()[0]
        
        predicted_class = int(np.argmax(probs))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return predicted_class, probs, float(entropy)
    
    def compare_methods(
        self,
        sequence: str,
        methods: list[Literal["integrated_gradients", "saliency", "attention", "gradcam", "lrp"]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Compare attribution methods side by side.
        
        Args:
            sequence: Input sequence
            methods: Methods to compare
            
        Returns:
            Dictionary mapping method name to attribution scores
        """
        if methods is None:
            methods = ["integrated_gradients", "saliency", "attention", "gradcam"]
        
        results = {}
        
        for method in methods:
            try:
                if method == "integrated_gradients":
                    ig_result = self.integrated_gradients(sequence)
                    results[method] = ig_result.position_importance
                elif method == "saliency":
                    results[method] = self.saliency_map(sequence)
                elif method == "attention":
                    attn = self.analyze_attention(sequence)
                    results[method] = attn.position_importance
                elif method == "gradcam":
                    gc = self.gradcam(sequence)
                    if gc.heatmaps:
                        results[method] = np.mean(list(gc.heatmaps.values()), axis=0)
                elif method == "lrp":
                    results[method] = self.lrp(sequence)
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                results[method] = np.zeros(len(sequence))
        
        return results


# ============================================================================
# Simple CNN Model (Compatible with saved weights)
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture matching saved weights structure.
    
    Architecture:
        Input (B, 4, L) -> Conv1 -> BN1 -> ReLU -> MaxPool
                       -> Conv2 -> BN2 -> ReLU -> MaxPool
                       -> Conv3 -> BN3 -> ReLU -> GlobalAvgPool
                       -> FC1 -> ReLU -> FC2 -> ReLU -> FC3
    """
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(4, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Handle input format: (B, L, 4) -> (B, 4, L)
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)
        
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_conv_features(self, x: Tensor) -> dict[str, Tensor]:
        """Get intermediate conv features for interpretation."""
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)
        
        features = {}
        
        x = self.conv1(x)
        features['conv1'] = x.clone()
        x = F.relu(self.bn1(x))
        x = self.pool(x)
        
        x = self.conv2(x)
        features['conv2'] = x.clone()
        x = F.relu(self.bn2(x))
        x = self.pool(x)
        
        x = self.conv3(x)
        features['conv3'] = x.clone()
        
        return features


# ============================================================================
# Model Loading Utilities
# ============================================================================

class ClassificationView(nn.Module):
    """
    Presents a trained checkpoint as a plain (batch, 3) classifier over
    prokaryote/eukaryote/virus, so every attribution method can index
    `output[:, target_class]` with the API's class indices.

    Needed because (a) RL agents return tuples such as (logits, value), and
    contrastive encoders return a normalized projection instead of class scores;
    (b) fine-grained models (8 taxonomic classes) must be rolled up to the three
    classes the API explains — otherwise class index 1 would silently mean
    "archaea" instead of "eukaryotic".
    """

    def __init__(self, model: nn.Module, class_names: list[str], kind: str, algorithm: str | None = None):
        super().__init__()
        from metapathpredict.config.settings import SUPERCLASSES, superclass_index_map

        self.model = model
        self.kind = kind
        self.algorithm = algorithm
        groups = torch.zeros(len(class_names), len(SUPERCLASSES))
        for i, j in enumerate(superclass_index_map(class_names)):
            groups[i, j] = 1.0
        self.register_buffer("groups", groups)

    def _class_probs(self, x: Tensor) -> Tensor:
        if self.kind == "contrastive":
            return F.softmax(self.model.encoder(x), dim=-1)
        out = self.model(x)
        if self.algorithm == "policy_gradient":
            return out[0]
        if self.algorithm == "dqn":
            return F.softmax(out, dim=-1)
        return F.softmax(out[0], dim=-1)  # actor_critic: (logits, value)

    def forward(self, x: Tensor) -> Tensor:
        # The interpreter encodes sequences as (batch, length, 4); the CNNs take (batch, 4, length).
        if x.dim() == 3 and x.shape[-1] == 4 and x.shape[1] != 4:
            x = x.transpose(1, 2)
        return torch.log((self._class_probs(x) @ self.groups).clamp_min(1e-8))

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@lru_cache(maxsize=4)
def load_model_for_interpretation(
    weights_path: str,
    device: str = "cpu",
    seq_length: int = 1000,
) -> ModelInterpreter:
    """
    Load model and create interpreter.

    Supports contrastive encoder and RL agent checkpoints.
    Uses caching to avoid reloading for repeated calls.
    """
    from metapathpredict.models import (
        ActorCriticAgent,
        ContrastiveEncoder,
        DQNAgent,
        PolicyGradientAgent,
    )

    weights_path = Path(weights_path)

    if not weights_path.exists():
        logger.warning(f"Weights not found at {weights_path}, using random ContrastiveEncoder for demo")
        model = ContrastiveEncoder(in_channels=4, backbone="small")
        return ModelInterpreter(model, device=device, seq_length=seq_length)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        logger.warning("Unknown checkpoint format, using random ContrastiveEncoder")
        model = ContrastiveEncoder(in_channels=4, backbone="small")
        return ModelInterpreter(model, device=device, seq_length=seq_length)

    config = checkpoint.get("config", {})

    if "encoder_state_dict" in checkpoint:
        # Contrastive encoder checkpoint
        model = ContrastiveEncoder(
            in_channels=4,
            backbone=config.get("backbone", "medium"),
            projection_dim=config.get("projection_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            base_channels=config.get("base_channels", 64),
            num_classes=checkpoint.get("num_classes", 3),
        )
        model.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
        model = ClassificationView(
            model, checkpoint.get("class_names", ["bacteria", "eukaryotic", "virus"]), "contrastive"
        )
        logger.info(f"Loaded ContrastiveEncoder from {weights_path}")

    elif "agent_state_dict" in checkpoint:
        # RL agent checkpoint
        algorithm = checkpoint.get("algorithm", "actor_critic")
        agent_cls = {
            "dqn": DQNAgent,
            "policy_gradient": PolicyGradientAgent,
            "actor_critic": ActorCriticAgent,
        }.get(algorithm, ActorCriticAgent)

        model = agent_cls(
            in_channels=4,
            num_actions=checkpoint.get("num_classes", 3),
            backbone=config.get("backbone", "medium"),
            hidden_dim=config.get("hidden_dim", 256),
            base_channels=checkpoint.get("base_channels", 64),
        )
        model.load_state_dict(checkpoint["agent_state_dict"], strict=False)
        model = ClassificationView(
            model, checkpoint.get("class_names", ["bacteria", "eukaryotic", "virus"]), "rl", algorithm
        )
        logger.info(f"Loaded {algorithm} agent from {weights_path}")

    elif "model_state_dict" in checkpoint:
        # Legacy CNN checkpoint
        model = SimpleCNN(num_classes=3)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Loaded legacy SimpleCNN from {weights_path}")

    else:
        logger.warning(f"Unknown checkpoint keys: {list(checkpoint.keys())}, using random model")
        model = ContrastiveEncoder(in_channels=4, backbone="small")

    return ModelInterpreter(model, device=device, seq_length=seq_length)
