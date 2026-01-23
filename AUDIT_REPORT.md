# MetaPathPredict - Отчёт об аудите

## Дата: 23 января 2026

## Результаты тестового запуска

```
========== Статистика тестов ==========
Всего тестов: 188
Прошло: 89 (47%)
Провалено: 89 (47%)
Пропущено: 2 (1%)
Ошибки импорта: 8 (4%)
```

## Критические проблемы

### 1. Несоответствие API между тестами и кодом

Тесты написаны для одного API, а код реализует другой:

| Тест ожидает | Код реализует |
|--------------|---------------|
| `MultiScaleCNN(in_channels=4)` | `MultiScaleCNN(seq_length=1000)` |
| `model.hidden_channels` | `model.branch_channels` |
| `config.fragment_size` | `config.default_fragment_size` |
| `config.epochs` | `config.num_epochs` |
| `config.train_split` | `config.train_ratio` |
| `ConfigurableCNN(preset="small")` | `ConfigurableCNN(kernel_preset="small")` |

### 2. Отсутствующие функции в preprocessing

**Исправлено**: Добавлены standalone функции:
- `clean_sequence()`
- `validate_sequence()`
- `fragment_sequence()`
- `get_reverse_complement()`

### 3. ConvBlock без pool_size

**Исправлено**: Добавлен опциональный параметр `pool_size` в ConvBlock.

### 4. KERNEL_PRESETS не экспортировался

**Исправлено**: Экспортирован на уровне модуля.

### 5. Pydantic warning о model_type

**Исправлено**: Добавлен `model_config = {"protected_namespaces": ()}` в ModelConfig.

## Файлы, требующие исправления тестов

| Файл | Количество провалов | Причина |
|------|---------------------|---------|
| `test_trainer.py` | 11 | Неправильный API моделей |
| `test_reinforcement.py` | 21 | Неправильный API агентов |
| `test_contrastive.py` | 18 | Неправильный API моделей |
| `test_integration.py` | 10 | Комплексное несоответствие |
| `test_config.py` | 9 | Неправильные атрибуты конфига |
| `test_preprocessing.py` | 12 | Неправильные сигнатуры функций |
| `test_tracking.py` | 8 | DuckDB не установлен (исправлено) |

## Исправленные проблемы

1. ✅ `ConvBlock` - добавлен `pool_size` параметр
2. ✅ `preprocessing.py` - добавлены standalone функции
3. ✅ `KERNEL_PRESETS` - экспортирован на уровне модуля
4. ✅ Pydantic warning - исправлено через `protected_namespaces`
5. ✅ DuckDB - установлен

## Рекомендации

### Краткосрочные (P0)

1. **Переписать тесты** под реальный API кода:
   - Обновить все вызовы `MultiScaleCNN` и `ResidualCNN`
   - Обновить все вызовы конфигурации
   - Обновить тесты reinforcement агентов

2. **Или обновить API кода** под тесты (менее предпочтительно)

### Среднесрочные (P1)

1. Добавить автоматическую генерацию тестов через pytest-generated
2. Использовать fixtures для создания моделей с правильным API
3. Добавить conftest.py с общими fixtures

### Долгосрочные (P2)

1. Внедрить contract testing между API и тестами
2. Добавить schema validation для конфигов
3. Генерировать документацию API из кода

## Команды для запуска

```bash
# Запуск всех тестов
pytest tests/ -v

# Запуск только проходящих тестов (models)
pytest tests/test_models.py -v

# Запуск конкретного теста
pytest tests/test_models.py::TestConfigurableCNN -v
```

## Актуальный API моделей

### MultiScaleCNN
```python
MultiScaleCNN(
    seq_length: int = 1000,
    num_classes: int = 3,
    kernel_sizes: list[int] = [5, 7, 11],
    branch_channels: int = 256,
    num_conv_layers: int = 3,
    hidden_dims: list[int] = None,
    dropout: float = 0.3,
    use_batch_norm: bool = True,
)
# Input: (B, L, 4) - batch, length, channels
# Output: (B, num_classes)
```

### ConfigurableCNN
```python
ConfigurableCNN(
    in_channels: int = 4,
    num_classes: int = 3,
    kernel_preset: Literal["small", "medium", "large", "progressive", "multi"] = "medium",
    custom_kernels: list[int] = None,
    base_channels: int = 64,
    num_blocks: int = 3,
    use_se: bool = True,
    dropout: float = 0.3,
)
# Input: (B, 4, L) - batch, channels, length
# Output: (B, num_classes)
```

### DataConfig
```python
DataConfig(
    fragment_sizes: list[int] = [500, 1000],
    default_fragment_size: int = 1000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
)
```

### TrainingConfig
```python
TrainingConfig(
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    # ... и другие параметры
)
```
