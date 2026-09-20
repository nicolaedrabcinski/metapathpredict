"""Prediction endpoints."""


from fastapi import APIRouter, HTTPException, Query

from metapathpredict.api.schemas import (
    ClassLabel,
    MethodPrediction,
    MethodType,
    PredictionComparison,
    PredictionRequest,
)
from metapathpredict.api.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()


@router.get("/{sample_id}", response_model=PredictionComparison)
async def get_sample_predictions(
    sample_id: int,
    methods: list[MethodType] | None = Query(
        None,
        description="Methods to include (default: all)"
    ),
):
    """
    Get predictions for a sample from all classification methods.

    Returns:
    - Predictions from Contrastive Learning and DRL methods
    - Consensus prediction (majority voting)
    - Agreement ratio between methods
    """
    result = await prediction_service.get_predictions(sample_id, methods)
    if not result:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    return result


@router.post("/predict", response_model=PredictionComparison)
async def predict_sequence(request: PredictionRequest):
    """
    Classify a new sequence using all methods.

    Accepts raw DNA sequence and returns predictions
    from all requested methods.
    """
    return await prediction_service.predict_sequence(
        sequence=request.sequence,
        methods=request.methods,
    )


@router.get("/{sample_id}/fragments", response_model=list[MethodPrediction])
async def get_fragment_predictions(
    sample_id: int,
    method: MethodType = Query(MethodType.CONTRASTIVE, description="Classification method"),
    fragment_size: int = Query(1000, ge=500, le=5000, description="Fragment size"),
):
    """
    Get predictions for each fragment of a sample.

    Useful for visualizing prediction consistency across sequence regions.
    """
    return await prediction_service.get_fragment_predictions(
        sample_id=sample_id,
        method=method,
        fragment_size=fragment_size,
    )


@router.get("/disagreements", response_model=list[PredictionComparison])
async def get_disagreements(
    min_disagreement: float = Query(0.5, ge=0, le=1, description="Min disagreement"),
    limit: int = Query(50, ge=1, le=200, description="Number of results"),
):
    """
    Find samples where methods disagree.

    Useful for identifying edge cases and potential labeling errors.
    """
    return await prediction_service.get_disagreements(
        min_disagreement=min_disagreement,
        limit=limit,
    )


@router.get("/by-class/{class_label}", response_model=list[PredictionComparison])
async def get_predictions_by_class(
    class_label: ClassLabel,
    method: MethodType = Query(MethodType.ENSEMBLE, description="Method for filtering"),
    confidence_min: float = Query(0.0, ge=0, le=1, description="Min confidence"),
    confidence_max: float = Query(1.0, ge=0, le=1, description="Max confidence"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Get samples predicted as a specific class.

    Useful for analyzing predictions by class.
    """
    return await prediction_service.get_by_class(
        class_label=class_label,
        method=method,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
        page=page,
        page_size=page_size,
    )
