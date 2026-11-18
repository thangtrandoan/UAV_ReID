# Hướng Dẫn Finetune GASNet Trên VeRi-776 và VRAI

Bộ công cụ `scripts/finetune_gasnet/` cung cấp 2 script độc lập để finetune nhẹ 3 mô hình GASNet (`resnet_rga`, `convnext_rga`, `convnext_bcgg`) từ các checkpoint gốc trong `weights/` sang **VeRi-776** và **VRAI**, đảm bảo:
1. **Không làm hỏng đặc trưng đã học (Anti-Catastrophic Forgetting):**
   * **Giai đoạn 1 (`--freeze-epochs 2`):** Đóng băng toàn bộ Backbone, Gate (`RGA`/`BCGG`), `OSBlockFS` và `BNNeck` (giữ nguyên thống kê BatchNorm ở chế độ `eval`), chỉ mở khóa 2 lớp phân loại mới (`classifier_global`, `classifier_fs`) để làm nóng (warmup) trọng số phân lớp theo số ID mới.
   * **Giai đoạn 2 (Full Finetune với Discriminative LR):** Mở khóa toàn mạng với 3 mức Learning Rate phân tầng:
     * `Backbone`: rất nhỏ (`1.5e-5` cho DINO-ConvNeXt, `3.5e-5` cho ResNet50-IBN)
     * `Gate + OSBlockFS + BNNeck`: trung bình (`1.0e-4`)
     * `Classifier Head`: cao (`3.5e-4`)
2. **Đánh giá Epoch 0 (Zero-shot Baseline) trước khi train:**
   * Tự động đo `mAP, Rank-1, Rank-5, Rank-10, mINP` ngay tại Epoch 0 và chỉ ghi đè file `.best.pth` khi `mAP` vượt qua kỷ lục trước đó.
3. **Chống rò rỉ dữ liệu tuyệt đối trên VRAI:**
   * Tự động tách đúng 1.000 ID của tập Validation (`load_vrai_val(seed=42)`) ra khỏi tập `images_train`, chỉ finetune trên **5.302 ID còn lại** và kiểm tra `Overlap = 0 IDs` trước khi chạy.

---

## 1. Cấu trúc thư mục

```text
scripts/finetune_gasnet/
├── __init__.py
├── common.py          # Engine finetune 2 giai đoạn, nạp weight khác số lớp, tối ưu hóa phân tầng LR
├── finetune_veri.py   # Script finetune độc lập trên VeRi-776 (576 train IDs -> 200 test IDs)
├── finetune_vrai.py   # Script finetune độc lập trên VRAI (5302 train IDs -> 1000 val IDs)
└── README.md          # Tài liệu hướng dẫn này
```

---

## 2. Chạy Finetune trên VeRi-776

### A. Finetune tự động cả 3 mô hình (Khuyến nghị để so sánh công bằng)
```bash
python3 -m scripts.finetune_gasnet.finetune_veri --variant all --epochs 15 --batch-size 64
```
Sau khi chạy xong, thư mục `weights/veri/` sẽ có đủ 3 checkpoint tốt nhất:
* `weights/veri/gasnet_resnet_rga.best.pth`
* `weights/veri/gasnet_convnext_rga.best.pth`
* `weights/veri/gasnet_convnext_bcgg.best.pth`
* Bảng tổng kết so sánh trước/sau finetune: `weights/veri/finetune_veri_summary.md` (và `.csv`).

### B. Hoặc finetune riêng từng mô hình
```bash
# 1. GASNet gốc (ResNet50-IBN + RGA)
python3 -m scripts.finetune_gasnet.finetune_veri --variant resnet_rga --epochs 15

# 2. GASNet + DINO-ConvNeXt + RGA
python3 -m scripts.finetune_gasnet.finetune_veri --variant convnext_rga --epochs 15

# 3. GASNet + DINO-ConvNeXt + BCGG
python3 -m scripts.finetune_gasnet.finetune_veri --variant convnext_bcgg --epochs 15
```

---

## 3. Chạy Finetune trên VRAI (5.302 Train IDs → 1.000 Val IDs)

### A. Finetune tự động cả 3 mô hình
```bash
python3 -m scripts.finetune_gasnet.finetune_vrai --variant all --epochs 12 --batch-size 64
```
Sau khi chạy xong, thư mục `weights/vrai/` sẽ có:
* `weights/vrai/gasnet_resnet_rga.best.pth`
* `weights/vrai/gasnet_convnext_rga.best.pth`
* `weights/vrai/gasnet_convnext_bcgg.best.pth`
* Bảng tổng kết: `weights/vrai/finetune_vrai_summary.md` (và `.csv`).

### B. Hoặc finetune riêng từng mô hình
```bash
python3 -m scripts.finetune_gasnet.finetune_vrai --variant resnet_rga --epochs 12
python3 -m scripts.finetune_gasnet.finetune_vrai --variant convnext_rga --epochs 12
python3 -m scripts.finetune_gasnet.finetune_vrai --variant convnext_bcgg --epochs 12
```

---

## 4. Chạy bộ Đánh giá & Trực quan hóa (`eval_gasnet`) với Weight đã Finetune

Bộ đánh giá `scripts.eval_gasnet.run_all` đã hỗ trợ cờ `--weights-dir` để trỏ thẳng vào thư mục chứa weight vừa finetune mà không làm mất weight gốc trong `weights/`:

### Đánh giá & Sinh ảnh Trực quan hóa với bộ weight đã finetune trên VeRi-776:
```bash
# Chạy Bảng 5.1, 5.2 và Visualize trên VeRi-776 dùng weight vừa finetune
python3 -m scripts.eval_gasnet.run_all \
    --weights-dir weights/veri \
    --output-dir outputs/gasnet_report_veri \
    --ablation-dataset veri \
    --vis-dataset veri \
    --all
```

### Đánh giá & Sinh ảnh Trực quan hóa với bộ weight đã finetune trên VRAI:
```bash
# Chạy Bảng 5.1, 5.2 và Visualize trên VRAI dùng weight vừa finetune
python3 -m scripts.eval_gasnet.run_all \
    --weights-dir weights/vrai \
    --output-dir outputs/gasnet_report_vrai \
    --ablation-dataset vrai \
    --vis-dataset vrai \
    --all
```

---

## 5. Các tham số tùy chỉnh thường dùng

| Tham số | Mặc định | Ý nghĩa |
| :--- | :---: | :--- |
| `--variant` | `all` | Chọn `resnet_rga`, `convnext_rga`, `convnext_bcgg`, hoặc `all` (chạy cả 3). |
| `--epochs` | `15` (VeRi) / `12` (VRAI) | Tổng số epoch finetune. |
| `--freeze-epochs` | `2` | Số epoch đầu đóng băng Backbone/Gate/BNNeck để làm nóng Classifier mới. |
| `--batch-size` | `64` | Kích thước batch PK (phải chia hết cho `--pk-k`, mặc định `64 = 16 IDs × 4 ảnh`). |
| `--backbone-lr` | Tự động (`1.5e-5` / `3.5e-5`) | Learning rate của tầng Backbone. |
| `--gate-lr` | `1.0e-4` | Learning rate của các khối Gate (`RGA`/`BCGG`), `OSBlockFS` và `BNNeck`. |
| `--head-lr` | `3.5e-4` | Learning rate của 2 lớp phân loại `classifier_global` và `classifier_fs`. |
| `--eval-every` | `3` | Chu kỳ đánh giá trên tập Test/Val sau mỗi N epoch. |
| `--strong-aug` | `False` | Bật thêm ColorJitter + RandomResizedCrop nếu muốn tăng cường dữ liệu mạnh hơn. |
| `--no-gem` | `False` | Dùng `AdaptiveAvgPool2d` thay cho `GeMPool (p=3.0)` (lưu thành `*_avgpool.best.pth` để không ghi đè bản GeM). |

Ví dụ finetune thêm bản **AvgPool** cho Bảng 5.4:
```bash
python3 -m scripts.finetune_gasnet.finetune_veri --variant convnext_bcgg --no-gem --epochs 15
python3 -m scripts.finetune_gasnet.finetune_vrai --variant convnext_bcgg --no-gem --epochs 12
```
