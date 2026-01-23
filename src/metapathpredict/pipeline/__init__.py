"""
Data Engineering pipeline integration.

Provides connectors for:
- DuckDB (OLAP analytics)
- DuckLake (lake format)
- MinIO/S3 (object storage)
- Ray (distributed computing)
"""

from .duckdb_connector import DuckDBConnector, DuckDBDataset
from .storage import S3Storage, MinIOStorage, ParquetWriter
from .ray_training import RayTrainer, distribute_training
from .dagster_assets import (
    prepare_dataset_asset,
    train_model_asset,
    predict_asset,
    evaluate_asset,
)

__all__ = [
    # DuckDB
    "DuckDBConnector",
    "DuckDBDataset",
    # Storage
    "S3Storage",
    "MinIOStorage", 
    "ParquetWriter",
    # Ray
    "RayTrainer",
    "distribute_training",
    # Dagster
    "prepare_dataset_asset",
    "train_model_asset",
    "predict_asset",
    "evaluate_asset",
]
