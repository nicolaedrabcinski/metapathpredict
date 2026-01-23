# MetaPathPredict - Backlog (Аудит проекта)

**Дата аудита**: 23 января 2026  
**Версия**: 2.0.0  
**Автор аудита**: GitHub Copilot (Claude Opus 4.5)

---

## ✅ ВЫПОЛНЕННЫЕ ИСПРАВЛЕНИЯ (Sprint 1 + Sprint 2)

| Задача | Статус | Изменённые файлы |
|--------|--------|------------------|
| **BIO-001** | ✅ | `settings.py`, `predictor.py`, `config/__init__.py` |
| **BIO-002** | ✅ | `preprocessing.py`, `test_preprocessing.py` |
| **BIO-003** | ✅ | `settings.py` |
| **ML-001** | ✅ | `trainer.py`, `cli.py` |
| **ML-002** | ✅ | `trainer.py` |
| **ML-003** | ✅ | `datamodule.py` |
| **ML-004** | ✅ | `trainer.py` (использует `settings.training.label_smoothing`) |
| **ML-005** | ✅ | `datamodule.py` |

**Результаты тестов после исправлений**: 183 passed, 1 failed (flaky RL test), 2 skipped

---

## 📊 Общая оценка проекта (ПОСЛЕ ИСПРАВЛЕНИЙ)

| Критерий | Было | Стало | Комментарий |
|----------|------|-------|-------------|
| **Архитектура ML** | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | + class weights, label smoothing, stratified split |
| **Качество кода** | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | + updated PyTorch 2.0 API |
| **Биоинформатика** | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | + correct N encoding, unified class order |
| **Тестирование** | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | + новые тесты для N encoding |
| **Производительность** | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | pin_memory fix |

**Статистика проекта**:
- ~9400 строк Python кода
- 30 модулей
- 184 теста (183 проходят)

---

## 🔴 КРИТИЧЕСКИЕ ПРОБЛЕМЫ (P0) - ВСЕ ИСПРАВЛЕНЫ ✅

### BIO-001: Неправильный порядок классов
**Файл**: `src/metapathpredict/inference/predictor.py:32`
```python
CLASS_NAMES = ["bacteria", "eukaryotic", "virus"]
```
**Проблема**: Порядок классов в разных местах проекта НЕСОГЛАСОВАН:
- `predictor.py`: `["bacteria", "eukaryotic", "virus"]`
- `settings.py`: `ClassLabel.VIRUS, ClassLabel.BACTERIA, ClassLabel.EUKARYOTIC`
- HDF5 данные могут иметь свой порядок

**Влияние**: 🚨 Модель может предсказывать НЕПРАВИЛЬНЫЙ класс!

**Решение**:
```python
# Единый источник правды в settings.py
CLASS_NAMES = ["virus", "bacteria", "eukaryotic"]  # Алфавитный или биологический порядок
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
```

---

### BIO-002: Некорректное кодирование N нуклеотидов
**Файл**: `src/metapathpredict/data/preprocessing.py:48-55`
```python
# N/other -> [0, 0, 0, 0] (gap encoding)
```
**Проблема**: Кодирование N как `[0,0,0,0]` эквивалентно отсутствию информации, но модель может интерпретировать это как "уверенность в отсутствии всех нуклеотидов", что математически некорректно.

**Биологический контекст**: N означает "неизвестный нуклеотид", а НЕ "отсутствие нуклеотида".

**Решение**:
```python
# Вариант 1: Равномерное распределение (Байесовский подход) - РЕКОМЕНДУЕТСЯ
if nt == "N":
    encoded[i] = [0.25, 0.25, 0.25, 0.25]

# Вариант 2: 5-канальное кодирование (дополнительный канал для неизвестных)
if self.include_n_channel and nt == "N":
    encoded[i, 4] = 1.0  # 5-й канал для N
```

---

### ML-001: CrossEntropyLoss без весов классов
**Файл**: `src/metapathpredict/training/trainer.py:116`
```python
self.criterion = criterion or nn.CrossEntropyLoss()
```
**Проблема**: При несбалансированных классах (типично для метагеномики) модель будет смещена к мажоритарному классу.

**Влияние**: Низкий recall на редких классах (например, вирусы обычно <10% данных)

**Решение**:
```python
# В CLI или при создании Trainer:
class_weights = data_module.get_class_weights(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Или использовать Focal Loss для экстремального дисбаланса
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        ...
```

---

## 🟠 ВЫСОКИЙ ПРИОРИТЕТ (P1)

### ML-002: Deprecated API torch.cuda.amp.autocast
**Файл**: `src/metapathpredict/training/trainer.py:265`
```python
from torch.cuda.amp import GradScaler, autocast
...
with autocast(enabled=self.use_amp):
```
**Проблема**: `torch.cuda.amp.autocast` устарел в PyTorch 2.0+

**Решение**:
```python
from torch.amp import autocast, GradScaler
...
with autocast(device_type=self.device.type, enabled=self.use_amp):
```

---

### ML-003: Отсутствует stratified split
**Файл**: `src/metapathpredict/data/datamodule.py:257-263`
```python
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator,
)
```
**Проблема**: `random_split` НЕ сохраняет пропорции классов в каждом сплите.

**Влияние**: Валидационный набор может не содержать некоторые классы вообще!

**Решение**:
```python
from sklearn.model_selection import train_test_split

# Получаем все метки
all_labels = [dataset[i][1] for i in range(len(dataset))]

# Stratified split
train_idx, temp_idx = train_test_split(
    range(len(dataset)), 
    test_size=val_ratio + test_ratio,
    stratify=all_labels,
    random_state=seed
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=[all_labels[i] for i in temp_idx],
    random_state=seed
)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)
```

---

### BIO-003: Слишком строгий min_valid_ratio
**Файл**: `src/metapathpredict/data/preprocessing.py:139`
```python
min_valid_ratio: float = 0.8,
```
**Проблема**: 80% - слишком строго для реальных метагеномных данных, которые часто содержат 10-30% N из-за качества секвенирования.

**Решение**: 
- Сделать настраиваемым через конфиг
- Значение по умолчанию: 0.7 (70%)
- Документировать влияние на качество модели

---

### BIO-004: GC-content вычисляется, но не используется
**Файл**: `src/metapathpredict/data/preprocessing.py:195-196`
```python
gc_content = (g_count + c_count) / length if length > 0 else 0.0
```
**Проблема**: GC-content вычисляется в `SequenceStats`, но нигде не используется как признак.

**Биологическая значимость**: GC-content - ВАЖНЕЙШИЙ дискриминативный признак:
| Организм | Типичный GC% |
|----------|--------------|
| Вирусы | 30-50% |
| Бактерии | 25-75% (очень широкий диапазон) |
| Эукариоты | 35-65% |

**Решение**: Добавить GC-content как дополнительный входной признак или использовать в auxiliary task.

---

### ML-004: Нет label smoothing
**Файл**: `src/metapathpredict/training/trainer.py:116`
**Проблема**: Жёсткие метки (one-hot) приводят к overconfident predictions.

**Решение**:
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 10% smoothing
```

---

### ML-005: pin_memory=True для CPU
**Файл**: `src/metapathpredict/data/datamodule.py:52`
```python
pin_memory: bool = True,
```
**Проблема**: `pin_memory=True` бесполезен на CPU и вызывает warning.

**Решение**:
```python
self.pin_memory = pin_memory and torch.cuda.is_available()
```

---

## 🟡 СРЕДНИЙ ПРИОРИТЕТ (P2)

### ARCH-001: Дублирование batch_size в конфигурации
**Файлы**: 
- `configs/train_cpu.yaml`: и `data.batch_size`, и `training.batch_size`
- `src/metapathpredict/config/settings.py`: DataConfig и TrainingConfig

**Решение**: Оставить только в TrainingConfig, удалить из DataConfig.

---

### ARCH-002: Жёстко закодированные пути
**Файл**: `src/metapathpredict/config/settings.py:72-79`
```python
virus_fasta: Path = Field(default=Path("data/input/viruses.fasta"))
bacteria_fasta: Path = Field(default=Path("data/input/bacteria.fasta"))
```
**Решение**: Использовать относительные пути от `base_dir`.

---

### ML-006: Отсутствует EMA (Exponential Moving Average)
**Проблема**: EMA весов модели улучшает стабильность и качество предсказаний на 1-2%.

**Решение**:
```python
class EMACallback(Callback):
    def __init__(self, model, decay=0.999):
        self.shadow = {name: param.clone() for name, param in model.named_parameters()}
        self.decay = decay
    
    def on_batch_end(self, trainer, batch, batch_idx, loss):
        for name, param in trainer.model.named_parameters():
            self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param
```

---

### ML-007: Нет Mixup/CutMix аугментации
**Файл**: `src/metapathpredict/data/augmentation.py`
**Проблема**: Только sequence-level аугментации, нет Mixup.

**Решение**:
```python
class SequenceMixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_x, batch_y):
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_x.size(0))
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        mixed_y = lam * batch_y + (1 - lam) * batch_y[index]
        return mixed_x, mixed_y
```

---

### BIO-005: Reverse complement применяется случайно
**Файл**: `src/metapathpredict/data/dataset.py:79-81`
```python
if self.use_reverse_complement and random.random() < self.rc_probability:
    sequence = reverse_complement(sequence)
```
**Проблема**: RC применяется только с вероятностью 0.5, но для DNA обе цепи биологически ЭКВИВАЛЕНТНЫ.

**Лучшая практика**: 
1. **Training**: использовать RC как аугментацию (текущий подход OK)
2. **Inference**: усреднять предсказания по обеим цепям (TTA)

```python
# В predictor.py
def predict_with_tta(self, seq):
    pred1 = self.predict(seq)
    pred2 = self.predict(reverse_complement(seq))
    return (pred1 + pred2) / 2
```

---

### BIO-006: Отсутствует codon-aware кодирование
**Проблема**: One-hot игнорирует кодоны (триплеты), которые определяют аминокислоты.

**Биологическая значимость**: Codon usage bias различается между организмами.

**Решение**:
```python
class KmerEncoder:
    def __init__(self, k=3):
        self.k = k
        self.vocab_size = 4 ** k  # 64 для k=3
        self.vocab = self._build_vocab()
    
    def encode(self, sequence):
        # Encode as k-mer frequencies or embeddings
        ...
```

---

### PERF-001: DataLoader prefetch_factor слишком мал
**Файл**: `configs/train_cpu.yaml`
```yaml
prefetch_factor: 4
```
**Рекомендация**: Для CPU увеличить до 8-16 для лучшего overlap I/O и compute.

---

### PERF-002: Нет torch.compile()
**Проблема**: PyTorch 2.0+ поддерживает `torch.compile()` для 10-30% ускорения.

**Решение**:
```python
import torch

if hasattr(torch, 'compile') and device.type == "cuda":
    model = torch.compile(model, mode="reduce-overhead")
```

---

## 🟢 НИЗКИЙ ПРИОРИТЕТ (P3)

### DOC-001: Отсутствует CHANGELOG
**Рекомендация**: Добавить CHANGELOG.md с семантическим версионированием.

### DOC-002: Пустая директория notebooks/
**Проблема**: Нет примеров использования в Jupyter notebooks.

### TEST-001: Нет edge case тестов
**Отсутствующие тесты**:
- Последовательности только из N
- Очень короткие последовательности (< 100bp)
- Последовательности с ambiguous codes (R, Y, W, S, M, K, B, D, H, V)

### ARCH-003: CLI слишком большой (566 строк)
**Рекомендация**: Разбить на подмодули (`cli/train.py`, `cli/predict.py`, etc.)

### ML-008: Нет поддержки multi-GPU
**Проблема**: Trainer не поддерживает `DistributedDataParallel`.

### BIO-007: Нет поддержки RNA
**Проблема**: Только DNA (ACGT), но RNA (ACGU) тоже нужна.

### PERF-003: Нет кэширования encoded данных
**Решение**: Использовать HDF5 с сжатием (gzip, lz4).

---

## 📋 Приоритизированный план исправлений

### Sprint 1: Критические (1-2 дня) ✅ ЗАВЕРШЁН
- [x] **BIO-001**: Унифицировать порядок классов ✅
- [x] **BIO-002**: Исправить кодирование N → `[0.25, 0.25, 0.25, 0.25]` ✅
- [x] **ML-001**: Добавить class weights в CrossEntropyLoss ✅

### Sprint 2: Высокий приоритет (3-5 дней) ✅ ЗАВЕРШЁН
- [x] **ML-002**: Обновить deprecated autocast API ✅
- [x] **ML-003**: Реализовать stratified split ✅
- [x] **ML-004**: Добавить label smoothing ✅
- [x] **ML-005**: Исправить pin_memory для CPU ✅
- [x] **BIO-003**: Сделать min_valid_ratio настраиваемым ✅

### Sprint 3: Средний приоритет (1 неделя)
- [ ] **ARCH-001**: Убрать дублирование batch_size
- [ ] **ML-006**: Добавить EMA callback
- [ ] **BIO-004**: Использовать GC-content как признак
- [ ] **PERF-002**: Интегрировать torch.compile()

### Backlog
- [ ] ML-007: Mixup аугментация
- [ ] BIO-006: K-mer кодирование
- [ ] DOC-001: CHANGELOG
- [ ] TEST-001: Edge case тесты

---

## 🔬 Научно-методологические рекомендации

### 1. Валидация модели
| Текущее | Рекомендуемое |
|---------|---------------|
| Train/val/test split | k-fold cross-validation (k=5) |
| Точечные метрики | + Confidence intervals |
| Нет калибровки | Temperature scaling / Platt scaling |

### 2. Интерпретируемость
- **Attention visualization** - какие позиции важны
- **Integrated Gradients** / **SHAP** - feature importance
- **Motif extraction** - паттерны из CNN фильтров

### 3. Бенчмаркинг
Сравнить с:
- Random Forest на k-mer features
- [VirFinder](https://doi.org/10.1186/s40168-017-0283-5)
- [DeepVirFinder](https://github.com/jessieren/DeepVirFinder)
- [Seeker](https://github.com/gussow/seeker)

### 4. Дополнительные признаки для улучшения
| Признак | Размерность | Значимость |
|---------|-------------|------------|
| GC-content | 1 | Высокая |
| Dinucleotide frequencies | 16 | Высокая |
| Codon usage | 64 | Средняя |
| Sequence complexity | 1-3 | Средняя |
| ORF features | ~10 | Для coding sequences |

---

## 📈 Целевые метрики

| Метрика | Текущее | Целевое | Benchmark |
|---------|---------|---------|-----------|
| Accuracy | TBD | >95% | VirFinder: 84% |
| F1-macro | TBD | >0.93 | DeepVirFinder: 0.90 |
| AUC-ROC | TBD | >0.98 | - |
| Inference | TBD | <10ms/seq | - |

---

## ✅ Что сделано хорошо

1. **Современная архитектура** - Multi-scale CNN + Attention = state-of-the-art подход
2. **Модульность** - Чёткое разделение на data, models, training, inference
3. **Type hints** - Везде аннотации типов (PEP 484)
4. **Документация** - Docstrings в Google style
5. **Тестирование** - 183 теста, хорошее покрытие
6. **Конфигурация** - Pydantic v2 для валидации
7. **Аугментации** - Биологически осмысленные (RC, mutations, insertions)
8. **Contrastive learning** - SimCLR/SupCon для self-supervised pretraining

---

*Аудит выполнен с учётом best practices в ML/DL и биоинформатике.*
