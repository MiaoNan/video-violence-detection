# 🔍 Violence Detection via Deep Spatiotemporal Models

This project focuses on detecting violent activities in videos using deep learning models such as **C3D**, **ConvLSTM**, and **DenseNet** variants. It supports various datasets including **CCTV footage**, **sports (hockey)**, and **movie clips**, and employs a modular training-validation pipeline with spatial-temporal augmentations.

---

## 🧠 Supported Models

- `VioNet_C3D`: 3D CNN for spatiotemporal feature extraction
- `VioNet_ConvLSTM`: CNN + LSTM hybrid for motion-aware modeling
- `VioNet_densenet`: DenseNet-based 2D CNN
- `VioNet_densenet_lean`: A lighter version of DenseNet

Only the `VioNet_densenet_lean` factory is included in this checkout, in `models/factory.py`; `model.py` re-exports it for compatibility. The original architecture backend and the other model implementations are missing.

---

## 🚀 How to Run

### 1. Set up your environment

```bash
pip install torch torchvision numpy opencv-python
```

> Training requires the missing original modules: `config`, `dataset`, `epoch`, `spatial_transforms`, `temporal_transforms`, `target_transforms`, and `utils`. Restore the DenseNet backend exposing `densenet88` and `get_fine_tuning_params`; its module name defaults to `densenet` and can be set with `config.densenet_backend`. Set `config.checkpoint_path` to compatible weights (default: `weights/weights.pth`). The included `cctv_weight.pth` has not been verified for compatibility.

### 2. Run the main training script

```bash
python main.py
```

You can change the model or dataset in the bottom section of `main.py`. For example:

```python
config = Config(
    'densenet_lean',  # or 'c3d', 'convlstm', 'densenet'
    'cctv',
    ...
)
```

---

## 📊 Logs & Checkpoints

Logs and model weights are saved to:

- `output/<run>/logs/*.log` — training/validation logs
- `output/<run>/weights/*.pth` — model weights

Example log files:
- `densenet_lean_fps16_vif_epoch5.log`
- `densenet_lean_fps16_vif_val5.log`

---

## 📦 Dataset Format

Datasets are loaded from `../VioDB/{dataset}_jpg/` with annotations in `../VioDB/{dataset}_jpg.json`.

- Folder structure expected: `VioDB/hockey_jpg`, `VioDB/cctv_jpg`, etc.
- JSON annotations specify clip labels.

---

## 📈 Sample Results

You can refer to included result visualizations:
- `Figure_cctv.png` – performance on CCTV dataset
- `loss_xiaorong.png` – loss curve
- `Ma_Baoguo_vs_Xu_Xiaodong_clip.mp4` – example input video

---

## 🧪 Cross Validation Support

Cross-validation is available with the following block:

```python
# for cv in range(1, 6):
#     config.num_cv = cv
#     main(config)
```

---

## 📍 Notes

- Logs are handled via a custom `Log` class for per-epoch and per-batch metrics.
- Spatial and temporal augmentations are modular (in `spatial_transforms`, `temporal_transforms`).
- Uses `torch.optim.lr_scheduler.ReduceLROnPlateau` to adapt learning rate.

---

## 🙋 Author

Michael He / MiaoNan
University of Victoria – Master of Data Science

## Code organization and dataset conversion

- `main.py`: training entry point and example configuration.
- `training/runner.py`: training, validation, scheduling, and checkpoint selection.
- `training/data.py`: data loaders and training/validation transforms.
- `models/factory.py`: model construction and device-aware weight loading.
- `preprocessing/cctv.py`: annotation loading and video-frame extraction.
- `cctv_json.py` and `model.py`: compatibility entry points/imports.

Convert the dataset with explicit paths:

```bash
python cctv_json.py --annotations /path/to/ground-truth.json \
  --video-dir /path/to/dataset
```

All videos are processed unless `--max-videos` is specified. Conversion keeps
this project's original first-annotation policy and 350-frame cap. Supporting
multiple annotated segments remains future work. Source subsets are preserved.
Frames are decoded from zero to align annotations correctly. Empty clips are
excluded from the generated database. Rerunning conversion replaces generated
`image_*.jpg` files and `n_frames` within each selected clip directory.
The Python conversion API now accepts annotation, video, and output paths
explicitly instead of using machine-specific global paths.

Training selects CUDA when available and otherwise CPU. Every run saves its
first validation checkpoint, then saves improvements in accuracy (using lower
loss to break ties); `acc_baseline` no longer gates saving. Existing checkpoint
formats and filenames are preserved. Worker count can be set with
`config.num_workers` (default: 4).

Run the extraction regression checks without PyTorch or OpenCV installed:

```bash
python3 -m unittest discover -s tests -v
```


## Run outputs

Each training call and dataset-conversion command creates a unique directory
under the project’s `output/` folder, using a UTC timestamp, run description,
and random suffix. `output/` is ignored by Git. The selected path is printed
at startup; training also exposes it as `config.run_dir` and returns it.

```text
output/
  <timestamp>_train_densenet_lean_cctv_cv0_<suffix>/
    logs/
    weights/
  <timestamp>_convert_cctv_<suffix>/
    cctv_jpg/
    cctv_jpg.json
```

Conversion's `--output-dir` and `--output-json` now select paths relative to its
new run directory; absolute paths and paths escaping that directory are rejected.
The lower-level `convert_cctv_dataset` function still accepts an explicit destination
for callers managing their own run. Point the training dataset paths at the
converted dataset when using conversion output. Pretrained checkpoints remain
inputs selected with `config.checkpoint_path`; saved checkpoints go into the
current training run's `weights/` folder. Existing tracked sample logs and weights
are retained in their original locations.
