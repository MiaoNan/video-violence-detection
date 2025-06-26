# 🔍 Violence Detection via Deep Spatiotemporal Models

This project focuses on detecting violent activities in videos using deep learning models such as **C3D**, **ConvLSTM**, and **DenseNet** variants. It supports various datasets including **CCTV footage**, **sports (hockey)**, and **movie clips**, and employs a modular training-validation pipeline with spatial-temporal augmentations.

---

## 🧠 Supported Models

- `VioNet_C3D`: 3D CNN for spatiotemporal feature extraction  
- `VioNet_ConvLSTM`: CNN + LSTM hybrid for motion-aware modeling  
- `VioNet_densenet`: DenseNet-based 2D CNN  
- `VioNet_densenet_lean`: A lighter version of DenseNet  

All models are defined in `model.py`.

---

## 🚀 How to Run

### 1. Set up your environment

```bash
pip install torch torchvision numpy
```

> (Also ensure dependencies like `spatial_transforms`, `temporal_transforms`, and `utils` are available)

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

- `./log/*.log` — training/validation logs  
- `./pth/*.pth` — model weights  

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