# Example configurations

This folder contains example configuration files for finetuning pyment models via the `pyment-finetune` CLI:

```
pyment-finetune <configuration.json>
```

All examples are written for the [ds000030](https://openneuro.org/datasets/ds000030) dataset. See the [main README](../README.md) for instructions on downloading and preprocessing it.

## Subfolders

- `local/` — configurations for local finetuning via the `pyment-finetune` CLI
- `docker/` — configurations for finetuning via Docker

## Examples

| File | Task | Model | Loss |
|------|------|-------|------|
| `local/ds000030_binary_diagnosis.json` | Binary classification (`has_diagnosis`) | `sfcn-bin` | `binary_crossentropy` |
| `local/ds000030_categorical_diagnosis.json` | 4-class classification (`diagnosis`) | `sfcn-cat` | `categorical_crossentropy` |
| `local/ds000030_regression_age.json` | Brain age regression | `sfcn-reg` | `mean_squared_error` |
| `docker/ds000030_binary_diagnosis.json` | Binary classification (`has_diagnosis`) | `sfcn-bin` | `binary_crossentropy` |
| `docker/ds000030_categorical_diagnosis.json` | 4-class classification (`diagnosis`) | `sfcn-cat` | `categorical_crossentropy` |
| `docker/ds000030_regression_age.json` | Brain age regression | `sfcn-reg` | `mean_squared_error` |

## Configuration reference

| Field | Required | Description |
|-------|----------|-------------|
| `dataset.images` | Yes | Path to the FastSurfer preprocessed folder |
| `dataset.labels` | Yes | Path to the labels CSV |
| `data_split.training_fraction` | Yes | Fraction of subjects used for training (remainder is validation) |
| `model.kind` | Yes | Model type: `sfcn-bin`, `sfcn-reg`, or `sfcn-cat` |
| `model.num_classes` | `sfcn-cat` only | Number of output classes |
| `model.input_shape` | No | Override the default input shape `(224, 192, 224)` |
| `model.dropout` | No | Dropout rate (default: `0.0`) |
| `target.name` | Yes | Column name in the labels CSV to predict |
| `target.kind` | Yes | Target type: `binary`, `categorical`, or `regression` |
| `target.labels` | Classification only | Ordered list of class values; determines the integer encoding |
| `pretrained_multitask_weights` | No | Pretrained weights identifier to transfer from (default: `multi-2025`) |
| `batch_size` | Yes | Training batch size |
| `num_threads` | No | Number of data loading threads (default: `1`) |
| `loss` | Yes | Keras loss function name |
| `metrics` | No | List of Keras metric names (default: `[]`) |
| `optimizer` | No | Keras optimizer name (default: `adam`) |
| `epochs` | Yes | Number of training epochs |
| `destination` | Yes | Output directory; receives `model.keras`, `history.json`, and `predictions.csv` |

Both `~` and environment variables (e.g. `$HOME`) are expanded in `dataset.images`, `dataset.labels`, and `destination`.

## Choosing the right `kind`

Both `model.kind` and `target.kind` use a discriminator field to select the correct variant. They must be compatible:

| Task | `model.kind` | `target.kind` | Recommended loss |
|------|-------------|---------------|-----------------|
| Binary classification | `sfcn-bin` | `binary` | `binary_crossentropy` |
| Multi-class classification | `sfcn-cat` | `categorical` | `categorical_crossentropy` |
| Regression | `sfcn-reg` | `regression` | `mean_squared_error` |
