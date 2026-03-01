"""Attribution and explainability endpoints for ML interpretability."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from metapathpredict.api.schemas import (
    AttributionResponse,
    AttributionRequest,
    AttributionMethod,
    MethodType,
    ClassLabel,
    MotifDiscoveryResponse,
    RegionAttribution,
)
from metapathpredict.api.services.attribution_service import AttributionService

router = APIRouter()
attribution_service = AttributionService()


@router.post("/compute", response_model=AttributionResponse)
async def compute_attribution(request: AttributionRequest):
    """
    Compute attribution scores for a sample using real ML interpretability methods.
    
    Methods:
    - saliency: Gradient-based attribution (SmoothGrad)
    - integrated_gradients: Axiomatic attribution (Sundararajan et al., 2017)
    - attention: Attention weight extraction from transformer layers
    - grad_cam: Activation-based attribution (Selvaraju et al., 2017)
    - lrp: Layer-wise Relevance Propagation
    
    Returns importance scores for each nucleotide position.
    """
    result = await attribution_service.compute(
        sample_id=request.sample_id,
        method=request.method,
        attribution_method=request.attribution_method,
        target_class=request.target_class,
        region_start=request.region_start,
        region_end=request.region_end,
        window_size=request.window_size,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Sample {request.sample_id} not found")
    return result


@router.get("/{sample_id}/saliency")
async def get_saliency_map(
    sample_id: int,
    method: MethodType = Query(MethodType.CONTRASTIVE, description="Classification method"),
    target_class: Optional[ClassLabel] = Query(None, description="Target class"),
    start: int = Query(0, ge=0, description="Start position"),
    end: Optional[int] = Query(None, description="End position"),
    smoothing: int = Query(5, ge=1, le=50, description="SmoothGrad samples (higher = smoother)"),
):
    """
    Get saliency map for sequence visualization.
    
    Uses SmoothGrad for noise-reduced gradient attribution.
    Returns smoothed attribution scores for D3.js visualization.
    """
    return await attribution_service.get_saliency_map(
        sample_id=sample_id,
        method=method,
        target_class=target_class,
        start=start,
        end=end,
        smoothing=smoothing,
    )


@router.get("/{sample_id}/regions", response_model=list[RegionAttribution])
async def get_important_regions(
    sample_id: int,
    method: MethodType = Query(MethodType.CONTRASTIVE, description="Classification method"),
    top_k: int = Query(10, ge=1, le=50, description="Number of top regions"),
    region_size: int = Query(50, ge=10, le=500, description="Region size in bp"),
):
    """
    Get most important sequence regions using Grad-CAM.
    
    Identifies contiguous regions that most influence model prediction.
    Returns top-k regions ranked by attribution importance.
    """
    return await attribution_service.get_important_regions(
        sample_id=sample_id,
        method=method,
        top_k=top_k,
        region_size=region_size,
    )


@router.get("/{sample_id}/motifs", response_model=MotifDiscoveryResponse)
async def discover_motifs(
    sample_id: int,
    method: MethodType = Query(MethodType.CONTRASTIVE, description="Classification method"),
    min_length: int = Query(6, ge=4, le=20, description="Min motif length"),
    max_length: int = Query(12, ge=6, le=30, description="Max motif length"),
    top_k: int = Query(10, ge=1, le=50, description="Number of motifs to return"),
):
    """
    Discover important sequence motifs from model activations.

    Extracts Position Weight Matrices (PWMs) from model layers that
    maximally respond to specific sequence patterns. These motifs
    represent learned features that influence classification.
    """
    return await attribution_service.discover_motifs(
        sample_id=sample_id,
        method=method,
        min_length=min_length,
        max_length=max_length,
        top_k=top_k,
    )


@router.get("/{sample_id}/compare-methods")
async def compare_attribution_methods(
    sample_id: int,
    methods: list[MethodType] = Query(
        [MethodType.CONTRASTIVE, MethodType.CONTRASTIVE],
        description="Methods to compare"
    ),
    region_start: Optional[int] = Query(None, description="Start of region"),
    region_end: Optional[int] = Query(None, description="End of region"),
):
    """
    Compare attributions from different interpretation methods.
    
    Shows which regions each method considers important, allowing
    identification of robust features (high across methods) vs
    method-specific artifacts.
    """
    return await attribution_service.compare_methods(
        sample_id=sample_id,
        methods=methods,
        region_start=region_start,
        region_end=region_end,
    )


@router.get("/{sample_id}/nucleotide-importance")
async def get_nucleotide_importance(
    sample_id: int,
    method: MethodType = Query(MethodType.CONTRASTIVE),
    position: int = Query(..., ge=0, description="Nucleotide position"),
    context_size: int = Query(50, ge=10, le=200, description="Context window size"),
):
    """
    Get detailed importance info for a specific nucleotide.
    
    Uses Integrated Gradients for precise attribution.
    
    Returns:
    - Attribution score for the position
    - Context sequence around the position
    - Context scores for visualization
    - Percentile ranking among context positions
    - Model prediction for context region
    """
    return await attribution_service.get_nucleotide_importance(
        sample_id=sample_id,
        method=method,
        position=position,
        context_size=context_size,
    )


@router.get("/{sample_id}/attention-analysis")
async def get_attention_analysis(
    sample_id: int,
    layer: Optional[int] = Query(None, ge=0, description="Specific layer to analyze (None for all)"),
):
    """
    Get detailed attention analysis for a sample.
    
    Extracts real attention weights from the trained transformer layers.
    
    Returns:
    - Per-layer attention patterns (which positions attend to which)
    - Per-head attention distributions  
    - Head specialization analysis:
      - Local heads: focus on nearby positions
      - Global heads: attend broadly across sequence
    - Attention entropy: measure of focus (low = focused, high = diffuse)
    - Attention sparsity: fraction of near-zero attention weights
    """
    result = await attribution_service.get_attention_analysis(
        sample_id=sample_id,
        layer=layer,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    return result


@router.get("/{sample_id}/interpretation-summary")
async def get_interpretation_summary(
    sample_id: int,
    target_class: Optional[ClassLabel] = Query(None, description="Target class to explain"),
):
    """
    Get comprehensive interpretation summary combining multiple methods.
    
    This endpoint aggregates results from:
    - Integrated Gradients: per-nucleotide importance
    - Attention Analysis: transformer attention patterns
    - Grad-CAM: Layer activations
    - Motif Discovery: learned sequence patterns
    
    Useful for generating full interpretability reports or
    understanding what the model learned about a sequence.
    """
    # Get attributions from multiple methods
    ig_result = await attribution_service.compute(
        sample_id=sample_id,
        method=MethodType.CONTRASTIVE,
        attribution_method=AttributionMethod.INTEGRATED_GRADIENTS,
        target_class=target_class,
        region_start=None,
        region_end=None,
        window_size=50,
    )
    
    attention_result = await attribution_service.get_attention_analysis(
        sample_id=sample_id,
        layer=None,
    )
    
    regions = await attribution_service.get_important_regions(
        sample_id=sample_id,
        method=MethodType.CONTRASTIVE,
        top_k=5,
        region_size=50,
    )
    
    motifs = await attribution_service.discover_motifs(
        sample_id=sample_id,
        method=MethodType.CONTRASTIVE,
        min_length=6,
        max_length=12,
        top_k=5,
    )
    
    return {
        "sample_id": sample_id,
        "target_class": target_class.value if target_class else "predicted",
        "integrated_gradients": {
            "top_positions": [
                {"position": p.position, "nucleotide": p.nucleotide, "score": p.score}
                for p in (ig_result.top_positions[:10] if ig_result else [])
            ],
            "num_regions": len(ig_result.regions) if ig_result else 0,
            "avg_importance": sum(r.mean_score for r in ig_result.regions) / len(ig_result.regions) if ig_result and ig_result.regions else 0,
        },
        "attention": {
            "num_layers": attention_result.get("num_layers", 0) if attention_result else 0,
            "entropy": attention_result.get("attention_entropy", []) if attention_result else [],
            "sparsity": attention_result.get("attention_sparsity", []) if attention_result else [],
            "has_data": bool(attention_result and attention_result.get("layers")),
        },
        "important_regions": [
            {
                "start": r.start, 
                "end": r.end, 
                "mean_score": r.mean_score,
                "sequence_preview": r.sequence[:20] + "..." if len(r.sequence) > 20 else r.sequence,
            }
            for r in regions[:5]
        ],
        "motifs": [
            {
                "pattern": m.pattern, 
                "consensus": m.consensus, 
                "importance": m.avg_importance,
                "occurrences": m.occurrences,
                "class_association": m.class_association.value,
            }
            for m in (motifs.motifs[:5] if motifs else [])
        ],
        "interpretation_methods": [
            "integrated_gradients",
            "attention_analysis", 
            "grad_cam",
            "motif_discovery",
        ],
    }
