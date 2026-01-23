# MetaPathPredict - Service Architecture

## Overview

MetaPathPredict использует современный стек Data Engineering для классификации метагеномных последовательностей ДНК. Проект построен на принципах MLOps и включает полный цикл от хранения данных до мониторинга.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MetaPathPredict Stack                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   MinIO     │  │  PostgreSQL │  │    Ray      │  │   Dagster   │        │
│  │  (Storage)  │  │  (Catalog)  │  │  (Compute)  │  │  (Orchest.) │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┼────────────────┼────────────────┘               │
│                          │                │                                 │
│                    ┌─────┴────────────────┴─────┐                          │
│                    │     MetaPathPredict App    │                          │
│                    │   (Training & Inference)   │                          │
│                    └────────────────────────────┘                          │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ Prometheus  │──│   Grafana   │  │   DataHub   │  (Optional)             │
│  │ (Metrics)   │  │   (Viz)     │  │ (Governance)│                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Services

### 1. MinIO (Object Storage)
**Роль:** S3-совместимое объектное хранилище для данных и артефактов

| Параметр | Значение |
|----------|----------|
| Порты | 9000 (API), 9001 (Console) |
| Image | `minio/minio:latest` |

**Buckets:**
- `metapathpredict` — основной bucket
- `metapathpredict-models` — сохранённые веса моделей
- `metapathpredict-datasets` — HDF5 датасеты
- `metapathpredict-artifacts` — MLflow/Dagster артефакты

**Зачем:**
- Хранение больших FASTA файлов и закодированных датасетов
- Версионирование моделей
- Интеграция с DuckDB через S3 протокол
- Масштабируемость — можно заменить на AWS S3/GCS в продакшене

---

### 2. PostgreSQL (Metadata Catalog)
**Роль:** Реляционная БД для метаданных, каталога и состояния пайплайнов

| Параметр | Значение |
|----------|----------|
| Порт | 5432 (GPU stack), 5433 (CPU stack) |
| Image | `postgres:15-alpine` |
| Database | `catalog`, `dagster` |

**Что хранит:**
- Каталог датасетов (DuckLake metadata)
- Состояние Dagster (runs, schedules, sensors)
- Метаданные экспериментов
- Lineage данных

**Зачем:**
- Единый источник истины для метаданных
- ACID транзакции для надёжности
- SQL интерфейс для аналитики

---

### 3. Ray (Distributed Compute)
**Роль:** Распределённые вычисления для обучения и инференса

| Компонент | Порты |
|-----------|-------|
| Ray Head | 8265 (Dashboard), 10001 (Client) |
| Ray Workers | Internal |

**Images:**
- GPU: `rayproject/ray:2.9.0-py310-gpu`
- CPU: `rayproject/ray:2.9.0-py310`

**Возможности:**
- **Ray Train** — распределённое обучение PyTorch моделей
- **Ray Tune** — гиперпараметрический поиск
- **Ray Data** — параллельная загрузка и предобработка данных
- **Ray Serve** — деплой моделей как микросервисов

**Зачем:**
- Горизонтальное масштабирование обучения
- Использование всех CPU/GPU в кластере
- Единый API для локальной и облачной разработки

---

### 4. Dagster (Orchestration)
**Роль:** Оркестрация ML пайплайнов

| Компонент | Порт |
|-----------|------|
| Webserver | 3000 |
| Daemon | — |

**Пайплайны:**
```python
# Основные jobs:
prepare_dataset_job    # FASTA → HDF5
train_model_job        # Training loop
predict_job            # Batch inference
full_pipeline_job      # End-to-end
```

**Зачем:**
- Декларативное описание пайплайнов
- Автоматический retry при ошибках
- Scheduling (по расписанию, по событию)
- Полная наблюдаемость (что, когда, как долго)
- Software-defined assets (датасеты как код)

---

## Optional Services

### 5. Prometheus (Metrics)
**Роль:** Сбор и хранение метрик

| Порт | Profile |
|------|---------|
| 9090 | `monitoring` |

**Метрики:**
- Training loss/accuracy по эпохам
- GPU/CPU utilization
- Memory consumption
- Inference latency (p50, p95, p99)

**Запуск:**
```bash
docker compose --profile monitoring up
```

---

### 6. Grafana (Visualization)
**Роль:** Визуализация метрик и алертинг

| Порт | Profile |
|------|---------|
| 3001 | `monitoring` |

**Dashboards:**
- Training Progress
- Resource Utilization
- Inference Performance

---

### 7. DataHub (Data Governance)
**Роль:** Каталог данных и lineage

| Порт | Profile |
|------|---------|
| 8080 | `datahub` |

**Зачем:**
- Документация датасетов
- Data lineage (откуда пришли данные)
- Quality checks
- Compliance (GDPR, etc.)

---

## Docker Compose Files

### `docker-compose.yml` (Full GPU Stack)
Полный стек для продакшена с GPU:
- Ray с GPU поддержкой
- Все сервисы включены
- Dagster orchestration

```bash
# Запуск
docker compose up -d

# С мониторингом
docker compose --profile monitoring up -d

# Все сервисы
docker compose --profile monitoring --profile datahub up -d
```

### `docker-compose.cpu.yml` (Minimal CPU Stack)
Минимальный стек для разработки:
- Только MinIO, PostgreSQL, App
- Без Ray (локальный PyTorch)
- Без оркестрации

```bash
# Запуск
docker compose -f docker-compose.cpu.yml up -d

# Training
docker compose -f docker-compose.cpu.yml run --rm app \
  python -m metapathpredict.cli train --config /app/configs/train_cpu.yaml
```

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                    │
└──────────────────────────────────────────────────────────────────────────┘

1. INPUT (FASTA files)
   │
   ▼
┌─────────────────┐
│  MinIO Bucket   │  s3://metapathpredict-datasets/raw/
│  (Raw Data)     │  - bacteria.fasta
└────────┬────────┘  - eukaryotic.fasta
         │           - viruses.fasta
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  Dagster: prepare_dataset_job
│  (CPU/Ray)      │  - Sequence validation
└────────┬────────┘  - One-hot encoding
         │           - Fragment generation
         │
         ▼
┌─────────────────┐
│  MinIO Bucket   │  s3://metapathpredict-datasets/encoded/
│  (HDF5)         │  - encoded_train.hdf5
└────────┬────────┘  - encoded_test.hdf5
         │
         │
         ▼
┌─────────────────┐
│  Training       │  Dagster: train_model_job
│  (GPU/Ray)      │  - PyTorch DataLoader
└────────┬────────┘  - CNN Model
         │           - Checkpointing
         │
         ▼
┌─────────────────┐
│  MinIO Bucket   │  s3://metapathpredict-models/
│  (Weights)      │  - best_model.pt
└────────┬────────┘  - checkpoints/
         │
         │
         ▼
┌─────────────────┐
│  Inference      │  Dagster: predict_job
│  (API/Batch)    │  - Load model
└────────┬────────┘  - Classify sequences
         │
         ▼
┌─────────────────┐
│  Output         │  s3://metapathpredict-artifacts/
│  (Results)      │  - predictions.csv
└─────────────────┘  - confusion_matrix.png
```

---

## Port Summary

| Service | Port | Description |
|---------|------|-------------|
| MinIO API | 9000 | S3-compatible API |
| MinIO Console | 9001 | Web UI |
| PostgreSQL | 5432/5433 | Database |
| Ray Dashboard | 8265 | Cluster monitoring |
| Ray Client | 10001 | Job submission |
| Dagster | 3000 | Pipeline UI |
| App | 8000/8888 | FastAPI server |
| Prometheus | 9090 | Metrics |
| Grafana | 3001 | Dashboards |
| DataHub | 8080 | Data catalog |

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-org/metapathpredict.git
cd metapathpredict

# 2. Start CPU stack (development)
docker compose -f docker-compose.cpu.yml up -d

# 3. Check services
docker compose -f docker-compose.cpu.yml ps

# 4. Run training
docker compose -f docker-compose.cpu.yml run --rm app \
  python -m metapathpredict.cli train \
  --config /app/configs/train_cpu.yaml \
  --epochs 10

# 5. View MinIO Console
open http://localhost:9001
# Login: minioadmin / minioadmin123
```

---

## Production Deployment

Для продакшена рекомендуется:

1. **Kubernetes** — использовать Helm charts вместо Docker Compose
2. **Managed Services:**
   - AWS S3 вместо MinIO
   - AWS RDS вместо PostgreSQL
   - AWS SageMaker вместо Ray
3. **Secrets Management** — HashiCorp Vault или AWS Secrets Manager
4. **CI/CD** — GitHub Actions + ArgoCD

См. [terraform/gcp/](../terraform/gcp/) для примера деплоя в GCP.
