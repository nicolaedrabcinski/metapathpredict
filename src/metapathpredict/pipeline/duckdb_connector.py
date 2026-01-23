"""
DuckDB connector for analytical queries and data storage.

DuckDB provides:
- Fast OLAP queries on sequence metadata
- Parquet file integration
- In-process analytics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class DuckDBConnector:
    """
    DuckDB connector for sequence data analytics.
    
    Supports:
    - Storing sequence metadata
    - Analytical queries
    - Parquet file operations
    - Integration with DuckLake
    """
    
    def __init__(
        self,
        database: str = ":memory:",
        read_only: bool = False,
    ):
        """
        Initialize DuckDB connection.
        
        Args:
            database: Path to database file or ":memory:".
            read_only: Whether to open in read-only mode.
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not installed. Run: pip install duckdb")
        
        self.database = database
        self.conn = duckdb.connect(database, read_only=read_only)
        
        # Enable extensions
        self._setup_extensions()
    
    def _setup_extensions(self) -> None:
        """Setup DuckDB extensions."""
        try:
            self.conn.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception:
            pass  # Extension may already be loaded
    
    def create_sequences_table(self) -> None:
        """Create table for sequence metadata."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sequences (
                id VARCHAR PRIMARY KEY,
                sequence_hash VARCHAR,
                length INTEGER,
                gc_content FLOAT,
                label VARCHAR,
                label_id INTEGER,
                source_file VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id VARCHAR PRIMARY KEY,
                sequence_id VARCHAR REFERENCES sequences(id),
                predicted_label VARCHAR,
                predicted_label_id INTEGER,
                confidence FLOAT,
                prob_bacteria FLOAT,
                prob_eukaryotic FLOAT,
                prob_virus FLOAT,
                model_version VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id VARCHAR PRIMARY KEY,
                model_type VARCHAR,
                config JSON,
                train_loss FLOAT,
                val_loss FLOAT,
                val_accuracy FLOAT,
                epochs INTEGER,
                duration_seconds FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def insert_sequence(
        self,
        seq_id: str,
        sequence: str,
        label: str,
        label_id: int,
        source_file: str = "",
    ) -> None:
        """Insert sequence metadata."""
        import hashlib
        
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()
        length = len(sequence)
        gc_content = (sequence.upper().count('G') + sequence.upper().count('C')) / length
        
        self.conn.execute("""
            INSERT OR REPLACE INTO sequences 
            (id, sequence_hash, length, gc_content, label, label_id, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [seq_id, seq_hash, length, gc_content, label, label_id, source_file])
    
    def insert_prediction(
        self,
        pred_id: str,
        sequence_id: str,
        predicted_label: str,
        predicted_label_id: int,
        confidence: float,
        probabilities: dict[str, float],
        model_version: str = "v2.0",
    ) -> None:
        """Insert prediction result."""
        self.conn.execute("""
            INSERT INTO predictions 
            (id, sequence_id, predicted_label, predicted_label_id, confidence,
             prob_bacteria, prob_eukaryotic, prob_virus, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            pred_id, sequence_id, predicted_label, predicted_label_id, confidence,
            probabilities.get("bacteria", 0),
            probabilities.get("eukaryotic", 0),
            probabilities.get("virus", 0),
            model_version,
        ])
    
    def log_training_run(
        self,
        run_id: str,
        model_type: str,
        config: dict,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        epochs: int,
        duration: float,
    ) -> None:
        """Log training run metadata."""
        self.conn.execute("""
            INSERT INTO training_runs
            (run_id, model_type, config, train_loss, val_loss, val_accuracy, epochs, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [run_id, model_type, json.dumps(config), train_loss, val_loss, val_accuracy, epochs, duration])
    
    def get_class_distribution(self) -> dict[str, int]:
        """Get class distribution from sequences table."""
        result = self.conn.execute("""
            SELECT label, COUNT(*) as count
            FROM sequences
            GROUP BY label
        """).fetchall()
        
        return {row[0]: row[1] for row in result}
    
    def get_prediction_accuracy(self, model_version: str | None = None) -> dict[str, Any]:
        """Calculate prediction accuracy metrics."""
        where_clause = f"WHERE p.model_version = '{model_version}'" if model_version else ""
        
        result = self.conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN s.label = p.predicted_label THEN 1 ELSE 0 END) as correct,
                AVG(p.confidence) as avg_confidence
            FROM predictions p
            JOIN sequences s ON p.sequence_id = s.id
            {where_clause}
        """).fetchone()
        
        total, correct, avg_conf = result
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "avg_confidence": avg_conf,
        }
    
    def export_to_parquet(self, output_path: str, table: str = "sequences") -> None:
        """Export table to Parquet file."""
        self.conn.execute(f"""
            COPY {table} TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    
    def import_from_parquet(self, input_path: str, table: str = "sequences") -> None:
        """Import data from Parquet file."""
        self.conn.execute(f"""
            INSERT INTO {table} SELECT * FROM read_parquet('{input_path}')
        """)
    
    def query(self, sql: str) -> list[tuple]:
        """Execute arbitrary SQL query."""
        return self.conn.execute(sql).fetchall()
    
    def close(self) -> None:
        """Close connection."""
        self.conn.close()


class DuckDBDataset(IterableDataset):
    """
    PyTorch dataset that reads from DuckDB with Parquet backend.
    
    Efficiently streams data for training without loading all into memory.
    """
    
    def __init__(
        self,
        parquet_path: str,
        preprocessor: Any,
        batch_size: int = 1000,
        shuffle: bool = True,
    ):
        """
        Initialize DuckDB-backed dataset.
        
        Args:
            parquet_path: Path to Parquet file with encoded sequences.
            preprocessor: Sequence preprocessor.
            batch_size: Batch size for reading.
            shuffle: Whether to shuffle data.
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not installed")
        
        self.parquet_path = parquet_path
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get total count
        conn = duckdb.connect()
        self.total_count = conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{parquet_path}')
        """).fetchone()[0]
        conn.close()
    
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over dataset."""
        conn = duckdb.connect()
        
        order_clause = "ORDER BY RANDOM()" if self.shuffle else ""
        
        # Stream in batches
        offset = 0
        while offset < self.total_count:
            batch = conn.execute(f"""
                SELECT encoded_sequence, label_id
                FROM read_parquet('{self.parquet_path}')
                {order_clause}
                LIMIT {self.batch_size}
                OFFSET {offset}
            """).fetchall()
            
            for row in batch:
                encoded = np.frombuffer(row[0], dtype=np.float32)
                # Reshape based on your encoding (e.g., 4 x length)
                encoded = encoded.reshape(4, -1)
                
                yield (
                    torch.from_numpy(encoded),
                    torch.tensor(row[1], dtype=torch.long),
                )
            
            offset += self.batch_size
        
        conn.close()
    
    def __len__(self) -> int:
        return self.total_count


class DuckLakeConnector:
    """
    DuckLake integration for lake-format data management.
    
    DuckLake provides:
    - ACID transactions
    - Schema evolution
    - Time travel
    - Metadata catalog (PostgreSQL)
    """
    
    def __init__(
        self,
        catalog_connection: str,
        storage_path: str,
    ):
        """
        Initialize DuckLake connection.
        
        Args:
            catalog_connection: PostgreSQL connection string for catalog.
            storage_path: S3/MinIO path for data storage.
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not installed")
        
        self.catalog_connection = catalog_connection
        self.storage_path = storage_path
        self.conn = duckdb.connect()
        
        self._setup_ducklake()
    
    def _setup_ducklake(self) -> None:
        """Setup DuckLake extension."""
        self.conn.execute("INSTALL ducklake; LOAD ducklake;")
        
        # Attach catalog
        self.conn.execute(f"""
            ATTACH '{self.catalog_connection}' AS catalog (TYPE DUCKLAKE);
        """)
    
    def create_table(
        self,
        table_name: str,
        schema: str,
    ) -> None:
        """Create DuckLake table."""
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS catalog.{table_name} (
                {schema}
            )
        """)
    
    def insert_sequences(
        self,
        table_name: str,
        data: list[dict],
    ) -> None:
        """Insert sequences into DuckLake table."""
        if not data:
            return
        
        columns = list(data[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_names = ", ".join(columns)
        
        self.conn.executemany(f"""
            INSERT INTO catalog.{table_name} ({column_names})
            VALUES ({placeholders})
        """, [tuple(d[c] for c in columns) for d in data])
    
    def query_as_of(
        self,
        table_name: str,
        timestamp: str,
        query: str = "*",
    ) -> list[tuple]:
        """Query table as of specific timestamp (time travel)."""
        return self.conn.execute(f"""
            SELECT {query}
            FROM catalog.{table_name}
            AS OF TIMESTAMP '{timestamp}'
        """).fetchall()
    
    def get_table_history(self, table_name: str) -> list[dict]:
        """Get table version history."""
        result = self.conn.execute(f"""
            SELECT * FROM ducklake_table_history('catalog', '{table_name}')
        """).fetchall()
        
        return [
            {"version": r[0], "timestamp": r[1], "operation": r[2]}
            for r in result
        ]
