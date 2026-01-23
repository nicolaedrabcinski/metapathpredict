"""
Object storage connectors for S3/MinIO.

Provides:
- Model artifact storage
- Dataset storage
- Parquet file management
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

try:
    import boto3
    from botocore.client import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger(__name__)


class S3Storage:
    """
    S3-compatible object storage client.
    
    Works with:
    - AWS S3
    - MinIO
    - AIStor
    - LocalStack
    """
    
    def __init__(
        self,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str = "us-east-1",
        bucket: str = "metapathpredict",
    ):
        """
        Initialize S3 client.
        
        Args:
            endpoint_url: S3 endpoint URL (for MinIO/AIStor).
            aws_access_key_id: Access key ID.
            aws_secret_access_key: Secret access key.
            region_name: AWS region.
            bucket: Default bucket name.
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 not installed. Run: pip install boto3")
        
        self.bucket = bucket
        
        # Get credentials from environment if not provided
        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=Config(signature_version="s3v4"),
        )
        
        self.resource = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=Config(signature_version="s3v4"),
        )
    
    def create_bucket(self, bucket: str | None = None) -> None:
        """Create bucket if not exists."""
        bucket = bucket or self.bucket
        
        try:
            self.client.head_bucket(Bucket=bucket)
        except Exception:
            self.client.create_bucket(Bucket=bucket)
            logger.info(f"Created bucket: {bucket}")
    
    def upload_file(
        self,
        local_path: str | Path,
        remote_key: str,
        bucket: str | None = None,
    ) -> str:
        """
        Upload file to S3.
        
        Args:
            local_path: Local file path.
            remote_key: S3 object key.
            bucket: Bucket name.
        
        Returns:
            S3 URI of uploaded file.
        """
        bucket = bucket or self.bucket
        
        self.client.upload_file(str(local_path), bucket, remote_key)
        
        uri = f"s3://{bucket}/{remote_key}"
        logger.info(f"Uploaded: {local_path} -> {uri}")
        
        return uri
    
    def download_file(
        self,
        remote_key: str,
        local_path: str | Path,
        bucket: str | None = None,
    ) -> Path:
        """
        Download file from S3.
        
        Args:
            remote_key: S3 object key.
            local_path: Local file path.
            bucket: Bucket name.
        
        Returns:
            Local path of downloaded file.
        """
        bucket = bucket or self.bucket
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client.download_file(bucket, remote_key, str(local_path))
        
        logger.info(f"Downloaded: s3://{bucket}/{remote_key} -> {local_path}")
        
        return local_path
    
    def upload_bytes(
        self,
        data: bytes,
        remote_key: str,
        bucket: str | None = None,
    ) -> str:
        """Upload bytes directly to S3."""
        bucket = bucket or self.bucket
        
        self.client.put_object(Bucket=bucket, Key=remote_key, Body=data)
        
        return f"s3://{bucket}/{remote_key}"
    
    def download_bytes(
        self,
        remote_key: str,
        bucket: str | None = None,
    ) -> bytes:
        """Download object as bytes."""
        bucket = bucket or self.bucket
        
        response = self.client.get_object(Bucket=bucket, Key=remote_key)
        return response["Body"].read()
    
    def upload_model(
        self,
        model: torch.nn.Module,
        remote_key: str,
        bucket: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Upload PyTorch model to S3.
        
        Args:
            model: PyTorch model.
            remote_key: S3 object key.
            bucket: Bucket name.
            metadata: Optional metadata to store alongside.
        
        Returns:
            S3 URI.
        """
        bucket = bucket or self.bucket
        
        # Save model to bytes
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        self.client.put_object(
            Bucket=bucket,
            Key=remote_key,
            Body=buffer.getvalue(),
        )
        
        # Save metadata
        if metadata:
            meta_key = remote_key.replace(".pt", "_metadata.json")
            self.client.put_object(
                Bucket=bucket,
                Key=meta_key,
                Body=json.dumps(metadata).encode(),
            )
        
        return f"s3://{bucket}/{remote_key}"
    
    def download_model(
        self,
        model: torch.nn.Module,
        remote_key: str,
        bucket: str | None = None,
    ) -> torch.nn.Module:
        """
        Download and load PyTorch model from S3.
        
        Args:
            model: Model instance to load weights into.
            remote_key: S3 object key.
            bucket: Bucket name.
        
        Returns:
            Model with loaded weights.
        """
        bucket = bucket or self.bucket
        
        response = self.client.get_object(Bucket=bucket, Key=remote_key)
        buffer = io.BytesIO(response["Body"].read())
        
        state_dict = torch.load(buffer, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
    
    def list_objects(
        self,
        prefix: str = "",
        bucket: str | None = None,
    ) -> list[dict]:
        """List objects in bucket."""
        bucket = bucket or self.bucket
        
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        return [
            {
                "key": obj["Key"],
                "size": obj["Size"],
                "modified": obj["LastModified"],
            }
            for obj in response.get("Contents", [])
        ]
    
    def delete_object(
        self,
        remote_key: str,
        bucket: str | None = None,
    ) -> None:
        """Delete object from S3."""
        bucket = bucket or self.bucket
        self.client.delete_object(Bucket=bucket, Key=remote_key)


class MinIOStorage(S3Storage):
    """
    MinIO-specific storage client.
    
    Convenience class with MinIO defaults.
    """
    
    def __init__(
        self,
        endpoint_url: str = "http://localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket: str = "metapathpredict",
        secure: bool = False,
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint_url: MinIO endpoint.
            access_key: Access key.
            secret_key: Secret key.
            bucket: Default bucket.
            secure: Whether to use HTTPS.
        """
        if secure and endpoint_url.startswith("http://"):
            endpoint_url = endpoint_url.replace("http://", "https://")
        
        super().__init__(
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            bucket=bucket,
        )


class ParquetWriter:
    """
    Parquet file writer for sequence data.
    
    Supports writing to:
    - Local filesystem
    - S3/MinIO
    """
    
    def __init__(
        self,
        storage: S3Storage | None = None,
        compression: str = "zstd",
    ):
        """
        Initialize Parquet writer.
        
        Args:
            storage: Optional S3 storage for remote writes.
            compression: Compression algorithm.
        """
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow not installed. Run: pip install pyarrow")
        
        self.storage = storage
        self.compression = compression
    
    def write_sequences(
        self,
        output_path: str,
        sequences: list[dict],
        partition_by: list[str] | None = None,
    ) -> str:
        """
        Write sequences to Parquet file.
        
        Args:
            output_path: Output path (local or s3://).
            sequences: List of sequence dictionaries.
            partition_by: Columns to partition by.
        
        Returns:
            Output path.
        """
        # Convert to PyArrow table
        table = pa.Table.from_pylist(sequences)
        
        if output_path.startswith("s3://"):
            # Write to S3
            bucket, key = output_path[5:].split("/", 1)
            
            buffer = io.BytesIO()
            pq.write_table(
                table,
                buffer,
                compression=self.compression,
            )
            buffer.seek(0)
            
            if self.storage:
                self.storage.upload_bytes(buffer.getvalue(), key, bucket)
            else:
                raise ValueError("S3 storage not configured")
        else:
            # Write locally
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if partition_by:
                pq.write_to_dataset(
                    table,
                    output_path,
                    partition_cols=partition_by,
                    compression=self.compression,
                )
            else:
                pq.write_table(
                    table,
                    output_path,
                    compression=self.compression,
                )
        
        return output_path
    
    def write_predictions(
        self,
        output_path: str,
        sequence_ids: list[str],
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: list[str],
    ) -> str:
        """
        Write predictions to Parquet.
        
        Args:
            output_path: Output path.
            sequence_ids: Sequence identifiers.
            predictions: Predicted class indices.
            probabilities: Class probabilities.
            class_names: Class names.
        
        Returns:
            Output path.
        """
        records = []
        
        for i, seq_id in enumerate(sequence_ids):
            record = {
                "sequence_id": seq_id,
                "predicted_class": int(predictions[i]),
                "predicted_label": class_names[predictions[i]],
                "confidence": float(probabilities[i].max()),
            }
            
            for j, name in enumerate(class_names):
                record[f"prob_{name}"] = float(probabilities[i, j])
            
            records.append(record)
        
        return self.write_sequences(output_path, records)
    
    def read_sequences(
        self,
        input_path: str,
        columns: list[str] | None = None,
    ) -> list[dict]:
        """
        Read sequences from Parquet.
        
        Args:
            input_path: Input path.
            columns: Columns to read (all if None).
        
        Returns:
            List of sequence dictionaries.
        """
        if input_path.startswith("s3://"):
            if not self.storage:
                raise ValueError("S3 storage not configured")
            
            bucket, key = input_path[5:].split("/", 1)
            data = self.storage.download_bytes(key, bucket)
            buffer = io.BytesIO(data)
            table = pq.read_table(buffer, columns=columns)
        else:
            table = pq.read_table(input_path, columns=columns)
        
        return table.to_pylist()
