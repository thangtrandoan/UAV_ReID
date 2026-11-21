# Hướng Dẫn Finetune GASNet Bằng File Cấu Hình YAML (`configs/gasnet_finetune.yaml`)

Hệ thống finetune sử dụng chung cấu trúc file cấu hình YAML giống hệt lúc train ([configs/gasnet_finetune.yaml](file:///home/nvidia/thang/UAV_ReID/configs/gasnet_finetune.yaml)). Bạn chỉ cần sửa trực tiếp trong file YAML (hoặc ghi đè nhanh qua `--set` trên dòng lệnh) các trường:
* `paths.data_root`: Thư mục chứa dữ liệu (`data`)
* `paths.weight_path`: Đường dẫn file weight gốc nạp vào để finetune (`weights/gasnet_convnext_bcgg.best.pth`, `weights/gasnet_convnext_rga.best.pth`, `weights/gasnet_resnet_rga.best.pth`)
* `paths.save_path`: Đường dẫn lưu weight sau khi finetune (ví dụ `weights/veri/gasnet_convnext_bcgg.best.pth`)
* `model.backbone`: `dinov3_convnext` | `resnet50_ibn` | `resnet50` | `swin_t`
* `model.gate`: `bcgg` | `rga`
* `model.use_gem`: `true` (GeMPool p=3.0) | `false` (AvgPool cho Bảng 5.4)
* `train.dataset`: `veri` | `vrai` | `vru` | `vehicleid`

---

## 1. Cấu trúc File Cấu Hình `configs/gasnet_finetune.yaml`

```yaml
paths:
  data_root: data
  weight_path: weights/gasnet_convnext_bcgg.best.pth
  save_path: weights/veri/gasnet_convnext_bcgg.best.pth
  log_path: outputs/gasnet_finetune/log.txt

model:
  backbone: dinov3_convnext                  # resnet50_ibn | resnet50 | swin_t | dinov3_convnext
  econvnext: true
  pretrained: false
  gate: bcgg                                 # bcgg | rga
  use_gem: true                              # true (GeM) | false (AvgPool)
  gem_p: 3.0

train:
  dataset: veri                              # veri | vrai | vru
  epochs: 15
  freeze_epochs: 2                           # Đóng băng Backbone/Gate/BNNeck 2 epoch đầu để warmup Classifier mới
  batch_size: 64
  pk_k: 4
  backbone_lr: 1.5e-5                        # LR nhỏ cho Backbone
  gate_lr: 1.0e-4                            # LR vừa cho Gate + OSBlockFS + BNNeck
  head_lr: 3.5e-4                            # LR lớn hơn cho Classifier mới
```

---

## 2. Cách Chạy Finetune Theo File Config (Giống Hệt Lúc Train)

### Cách A — Sửa trực tiếp trong `configs/gasnet_finetune.yaml` rồi chạy:
Bạn có thể gọi qua `scripts.finetune_gasnet` hoặc dùng chính `scripts/train_gasnet.py`:
```bash
python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml
```
*(Hoặc: `python3 scripts/train_gasnet.py --config configs/gasnet_finetune.yaml` đều chạy chung engine này).*

---

### Cách B — Giữ nguyên 1 file config và đổi nhanh bằng `--set` cho từng Model & Dataset:

#### 1. Finetune trên **VeRi-776** (`train.dataset=veri`):
* **Model 1: DINO-ConvNeXt + BCGG (GeM):**
  ```bash
  python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
      --set train.dataset=veri \
            paths.weight_path=weights/gasnet_convnext_bcgg.best.pth \
            paths.save_path=weights/veri/gasnet_convnext_bcgg.best.pth \
            model.backbone=dinov3_convnext model.gate=bcgg model.use_gem=true
  ```
* **Model 2: DINO-ConvNeXt + RGA:**
  ```bash
  python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
      --set train.dataset=veri \
            paths.weight_path=weights/gasnet_convnext_rga.best.pth \
            paths.save_path=weights/veri/gasnet_convnext_rga.best.pth \
            model.backbone=dinov3_convnext model.gate=rga model.use_gem=true
  ```
* **Model 3: ResNet50-IBN + RGA:**
  ```bash
  python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
      --set train.dataset=veri \
            paths.weight_path=weights/gasnet_resnet_rga.best.pth \
            paths.save_path=weights/veri/gasnet_resnet_rga.best.pth \
            model.backbone=resnet50_ibn model.gate=rga model.use_gem=true \
            train.backbone_lr=3.5e-5
  ```
* **Model 4 (Cho Bảng 5.4): DINO-ConvNeXt + BCGG dùng AvgPool (`model.use_gem=false`):**
  ```bash
  python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
      --set train.dataset=veri \
            paths.weight_path=weights/gasnet_convnext_bcgg.best.pth \
            paths.save_path=weights/veri/gasnet_convnext_bcgg_avgpool.best.pth \
            model.backbone=dinov3_convnext model.gate=bcgg model.use_gem=false
  ```

---

#### 2. Finetune trên **VRAI** (`train.dataset=vrai` — Tự động tách 5.302 Train IDs / 1.000 Val IDs không trùng lặp):
* **Model 1: DINO-ConvNeXt + BCGG:**
  ```bash
  python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
      --set train.dataset=vrai train.epochs=12 \
            paths.weight_path=weights/gasnet_convnext_bcgg.best.pth \
            paths.save_path=weights/vrai/gasnet_convnext_bcgg.best.pth \
            model.backbone=dinov3_convnext model.gate=bcgg
  ```
* **Model 2: DINO-ConvNeXt + RGA:**
  ```bash
  python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
      --set train.dataset=vrai train.epochs=12 \
            paths.weight_path=weights/gasnet_convnext_rga.best.pth \
            paths.save_path=weights/vrai/gasnet_convnext_rga.best.pth \
            model.backbone=dinov3_convnext model.gate=rga
  ```
* **Model 3: ResNet50-IBN + RGA:**
  ```bash
  python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
      --set train.dataset=vrai train.epochs=12 \
            paths.weight_path=weights/gasnet_resnet_rga.best.pth \
            paths.save_path=weights/vrai/gasnet_resnet_rga.best.pth \
            model.backbone=resnet50_ibn model.gate=rga \
            train.backbone_lr=3.5e-5
  ```

#### 3. Finetune trên **PKU-VehicleID V1.0** (`train.dataset=vehicleid` — Train trên 13,164 IDs; Valid trên Small 800 & Medium 1600; Chọn Best Checkpoint theo Medium 1600; Đánh giá cuối sau khi train xong trên Large 2400):
* **Chạy riêng script `finetune_vehicleid.py` (tất cả 3 biến thể hoặc từng biến thể):**
  ```bash
  python3 -m scripts.finetune_gasnet.finetune_vehicleid --variant all --epochs 15
  ```
* **Hoặc chạy bằng config YAML (`--set train.dataset=vehicleid`):**
  ```bash
  python3 -m scripts.finetune_gasnet --config configs/gasnet_finetune.yaml \
      --set train.dataset=vehicleid train.epochs=15 \
            paths.weight_path=weights/gasnet_convnext_bcgg.best.pth \
            paths.save_path=weights/vehicleid/gasnet_convnext_bcgg.best.pth \
            model.backbone=dinov3_convnext model.gate=bcgg
  ```

---

## 3. Đánh Giá & Sinh Ảnh Trực Quan Với Bộ Weight Vừa Finetune

Sau khi finetune xong vào `weights/veri/` hoặc `weights/vrai/`, bạn chạy lệnh đánh giá:

```bash
# Đánh giá trên VeRi-776 với weight đã finetune
python3 -m scripts.eval_gasnet.run_all \
    --weights-dir weights/veri \
    --output-dir outputs/gasnet_report_veri \
    --ablation-dataset veri \
    --vis-dataset veri \
    --all
```
