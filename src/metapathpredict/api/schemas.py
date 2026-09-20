"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ClassLabel(str, Enum):
    """Sequence class labels."""
    BACTERIA = "bacteria"
    EUKARYOTIC = "eukaryotic"
    VIRUS = "virus"


class MethodType(str, Enum):
    """Classification method types."""
    CONTRASTIVE = "contrastive"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"


# =============================================================================
# Sample schemas
# =============================================================================

class SampleBase(BaseModel):
    """Base sample schema."""
    ncbi_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    source: str = "unknown"
    organism: Optional[str] = None


class SampleCreate(SampleBase):
    """Schema for creating a sample."""
    sequence: str = Field(..., min_length=100, description="DNA sequence")


class SampleSummary(SampleBase):
    """Summary of a sample (without full sequence)."""
    id: int
    length: int
    gc_content: float
    true_label: Optional[ClassLabel] = None
    
    class Config:
        from_attributes = True


class SampleDetail(SampleSummary):
    """Full sample details including sequence."""
    sequence: str
    fragments_count: int = 0


class SampleListResponse(BaseModel):
    """Paginated list of samples."""
    items: list[SampleSummary]
    total: int
    page: int
    page_size: int
    pages: int


# =============================================================================
# Prediction schemas
# =============================================================================

class ClassProbabilities(BaseModel):
    """Probabilities for each class."""
    bacteria: float = Field(..., ge=0, le=1)
    eukaryotic: float = Field(..., ge=0, le=1)
    virus: float = Field(..., ge=0, le=1)


class MethodPrediction(BaseModel):
    """Prediction from a single method."""
    method: MethodType
    predicted_class: ClassLabel
    confidence: float = Field(..., ge=0, le=1)
    probabilities: ClassProbabilities


class PredictionComparison(BaseModel):
    """Comparison of predictions from all methods."""
    sample_id: int
    sample_name: str
    true_label: Optional[ClassLabel] = None
    predictions: list[MethodPrediction]
    consensus: Optional[ClassLabel] = None
    agreement: float = Field(..., ge=0, le=1, description="Agreement ratio between methods")


class PredictionRequest(BaseModel):
    """Request for prediction."""
    sequence: str = Field(..., min_length=100)
    methods: list[MethodType] = [MethodType.CONTRASTIVE, MethodType.REINFORCEMENT]


# =============================================================================
# Statistics schemas
# =============================================================================

class DatasetStats(BaseModel):
    """Dataset statistics."""
    total_samples: int
    by_class: dict[str, int]
    by_source: dict[str, int]
    avg_length: float
    avg_gc_content: float


class ModelStats(BaseModel):
    """Model performance statistics."""
    method: MethodType
    accuracy: float
    f1_score: float
    confusion_matrix: list[list[int]]
