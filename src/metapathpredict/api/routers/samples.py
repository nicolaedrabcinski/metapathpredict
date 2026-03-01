"""Sample management endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from metapathpredict.api.schemas import (
    SampleListResponse,
    SampleDetail,
    SampleSummary,
    SampleCreate,
    ClassLabel,
    DatasetStats,
)
from metapathpredict.api.services.sample_service import SampleService

router = APIRouter()
sample_service = SampleService()


@router.get("", response_model=SampleListResponse)
async def list_samples(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    class_filter: Optional[ClassLabel] = Query(None, description="Filter by class"),
    source_filter: Optional[str] = Query(None, description="Filter by source (e.g., 'NCBI')"),
    search: Optional[str] = Query(None, description="Search by name or NCBI ID"),
    min_length: Optional[int] = Query(None, ge=0, description="Minimum sequence length"),
    max_length: Optional[int] = Query(None, description="Maximum sequence length"),
    sort_by: str = Query("id", description="Sort field"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="Sort order"),
):
    """
    List all samples with filtering and pagination.
    
    Supports filtering by:
    - Class label (bacteria, eukaryotic, virus)
    - Source (NCBI, custom, etc.)
    - Sequence length range
    - Text search on name/NCBI ID
    """
    return await sample_service.list_samples(
        page=page,
        page_size=page_size,
        class_filter=class_filter,
        source_filter=source_filter,
        search=search,
        min_length=min_length,
        max_length=max_length,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats():
    """Get overall dataset statistics."""
    return await sample_service.get_stats()


@router.get("/{sample_id}", response_model=SampleDetail)
async def get_sample(sample_id: int):
    """
    Get detailed information about a specific sample.
    
    Includes full sequence and metadata.
    """
    sample = await sample_service.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    return sample


@router.get("/{sample_id}/sequence")
async def get_sample_sequence(
    sample_id: int,
    start: int = Query(0, ge=0, description="Start position (0-indexed)"),
    end: Optional[int] = Query(None, description="End position (exclusive)"),
):
    """
    Get sequence region for a sample.
    
    Useful for paginated sequence viewing.
    """
    result = await sample_service.get_sequence_region(sample_id, start, end)
    if not result:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    return result


@router.post("", response_model=SampleSummary, status_code=201)
async def create_sample(sample: SampleCreate):
    """
    Create a new sample from uploaded sequence.
    
    Automatically computes:
    - Sequence length
    - GC content
    - Fragment count
    """
    return await sample_service.create_sample(sample)


@router.delete("/{sample_id}", status_code=204)
async def delete_sample(sample_id: int):
    """Delete a sample."""
    deleted = await sample_service.delete_sample(sample_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")


@router.get("/{sample_id}/similar", response_model=list[SampleSummary])
async def get_similar_samples(
    sample_id: int,
    limit: int = Query(10, ge=1, le=50, description="Number of similar samples"),
):
    """
    Find samples similar to the given sample.
    
    Uses embedding similarity from contrastive model.
    """
    return await sample_service.get_similar_samples(sample_id, limit)
