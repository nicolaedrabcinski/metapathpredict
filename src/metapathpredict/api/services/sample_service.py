"""Sample management service."""

import math
from typing import Optional
from pathlib import Path

from metapathpredict.api.schemas import (
    SampleListResponse,
    SampleDetail,
    SampleSummary,
    SampleCreate,
    ClassLabel,
    DatasetStats,
)


def compute_gc_content(sequence: str) -> float:
    """Compute GC content of a sequence."""
    sequence = sequence.upper()
    gc_count = sequence.count("G") + sequence.count("C")
    total = len(sequence)
    return gc_count / total if total > 0 else 0.0


class SampleService:
    """Service for managing samples."""
    
    def __init__(self):
        # In-memory storage for demo (replace with database)
        self._samples: dict[int, dict] = {}
        self._next_id = 1
        self._load_demo_samples()
    
    def _load_demo_samples(self):
        """Load demo samples for development."""
        demo_samples = [
            {
                "ncbi_id": "NC_001416.1",
                "name": "Lambda phage",
                "description": "Enterobacteria phage lambda, complete genome",
                "source": "NCBI",
                "organism": "Enterobacteria phage lambda",
                "sequence": "ATGCGATCGATCGATCGATCG" * 500,  # Demo sequence
                "true_label": ClassLabel.VIRUS,
            },
            {
                "ncbi_id": "NC_000913.3",
                "name": "E. coli K-12",
                "description": "Escherichia coli str. K-12 substr. MG1655",
                "source": "NCBI",
                "organism": "Escherichia coli",
                "sequence": "GCTAGCTAGCTAGCTAGCTA" * 500,
                "true_label": ClassLabel.BACTERIA,
            },
            {
                "ncbi_id": "NC_001144.5",
                "name": "S. cerevisiae Chr XII",
                "description": "Saccharomyces cerevisiae S288C chromosome XII",
                "source": "NCBI",
                "organism": "Saccharomyces cerevisiae",
                "sequence": "TACGTACGTACGTACGTACG" * 500,
                "true_label": ClassLabel.EUKARYOTIC,
            },
        ]
        
        for sample in demo_samples:
            self._create_sample_internal(sample)
    
    def _create_sample_internal(self, data: dict) -> SampleSummary:
        """Create sample internally."""
        sample_id = self._next_id
        self._next_id += 1
        
        sequence = data.get("sequence", "")
        
        self._samples[sample_id] = {
            "id": sample_id,
            "ncbi_id": data.get("ncbi_id"),
            "name": data["name"],
            "description": data.get("description"),
            "source": data.get("source", "unknown"),
            "organism": data.get("organism"),
            "sequence": sequence,
            "length": len(sequence),
            "gc_content": compute_gc_content(sequence),
            "true_label": data.get("true_label"),
            "fragments_count": max(1, len(sequence) // 1000),
        }
        
        return self._to_summary(self._samples[sample_id])
    
    def _to_summary(self, sample: dict) -> SampleSummary:
        """Convert sample dict to summary."""
        return SampleSummary(
            id=sample["id"],
            ncbi_id=sample.get("ncbi_id"),
            name=sample["name"],
            description=sample.get("description"),
            source=sample["source"],
            organism=sample.get("organism"),
            length=sample["length"],
            gc_content=sample["gc_content"],
            true_label=sample.get("true_label"),
        )
    
    def _to_detail(self, sample: dict) -> SampleDetail:
        """Convert sample dict to detail."""
        return SampleDetail(
            id=sample["id"],
            ncbi_id=sample.get("ncbi_id"),
            name=sample["name"],
            description=sample.get("description"),
            source=sample["source"],
            organism=sample.get("organism"),
            length=sample["length"],
            gc_content=sample["gc_content"],
            true_label=sample.get("true_label"),
            sequence=sample["sequence"],
            fragments_count=sample["fragments_count"],
        )
    
    async def list_samples(
        self,
        page: int = 1,
        page_size: int = 20,
        class_filter: Optional[ClassLabel] = None,
        source_filter: Optional[str] = None,
        search: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        sort_by: str = "id",
        sort_order: str = "asc",
    ) -> SampleListResponse:
        """List samples with filtering and pagination."""
        # Filter samples
        samples = list(self._samples.values())
        
        if class_filter:
            samples = [s for s in samples if s.get("true_label") == class_filter]
        
        if source_filter:
            samples = [s for s in samples if s["source"].lower() == source_filter.lower()]
        
        if search:
            search = search.lower()
            samples = [
                s for s in samples
                if search in s["name"].lower()
                or (s.get("ncbi_id") and search in s["ncbi_id"].lower())
            ]
        
        if min_length is not None:
            samples = [s for s in samples if s["length"] >= min_length]
        
        if max_length is not None:
            samples = [s for s in samples if s["length"] <= max_length]
        
        # Sort
        reverse = sort_order == "desc"
        samples.sort(key=lambda s: s.get(sort_by, 0), reverse=reverse)
        
        # Paginate
        total = len(samples)
        pages = math.ceil(total / page_size) if total > 0 else 1
        start = (page - 1) * page_size
        end = start + page_size
        page_samples = samples[start:end]
        
        return SampleListResponse(
            items=[self._to_summary(s) for s in page_samples],
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )
    
    async def get_sample(self, sample_id: int) -> Optional[SampleDetail]:
        """Get sample by ID."""
        sample = self._samples.get(sample_id)
        if not sample:
            return None
        return self._to_detail(sample)
    
    async def get_sequence_region(
        self,
        sample_id: int,
        start: int,
        end: Optional[int],
    ) -> Optional[dict]:
        """Get sequence region."""
        sample = self._samples.get(sample_id)
        if not sample:
            return None
        
        sequence = sample["sequence"]
        if end is None:
            end = len(sequence)
        
        return {
            "sample_id": sample_id,
            "start": start,
            "end": min(end, len(sequence)),
            "sequence": sequence[start:end],
            "total_length": len(sequence),
        }
    
    async def create_sample(self, data: SampleCreate) -> SampleSummary:
        """Create a new sample."""
        return self._create_sample_internal(data.model_dump())
    
    async def delete_sample(self, sample_id: int) -> bool:
        """Delete a sample."""
        if sample_id in self._samples:
            del self._samples[sample_id]
            return True
        return False
    
    async def get_similar_samples(
        self,
        sample_id: int,
        limit: int = 10,
    ) -> list[SampleSummary]:
        """Get similar samples (placeholder)."""
        # TODO: Implement using contrastive embeddings
        samples = [
            self._to_summary(s) 
            for sid, s in self._samples.items() 
            if sid != sample_id
        ]
        return samples[:limit]
    
    async def get_stats(self) -> DatasetStats:
        """Get dataset statistics."""
        samples = list(self._samples.values())
        
        by_class = {}
        by_source = {}
        total_length = 0
        total_gc = 0
        
        for s in samples:
            label = s.get("true_label")
            if label:
                label_str = label.value if hasattr(label, "value") else str(label)
                by_class[label_str] = by_class.get(label_str, 0) + 1
            
            source = s["source"]
            by_source[source] = by_source.get(source, 0) + 1
            
            total_length += s["length"]
            total_gc += s["gc_content"]
        
        n = len(samples) or 1
        
        return DatasetStats(
            total_samples=len(samples),
            by_class=by_class,
            by_source=by_source,
            avg_length=total_length / n,
            avg_gc_content=total_gc / n,
        )
