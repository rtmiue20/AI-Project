# ĐỀ TÀI S08: LoRA & QLoRA — TINH CHỈNH MÔ HÌNH LỚN HIỆU QUẢ VỀ SỐ LƯỢNG THAM SỐ

> **Học phần:** Trí tuệ Nhân tạo (*Artificial Intelligence*) — Đại học Kinh tế TP. Hồ Chí Minh (UEH)  
> **Quy mô nhóm:** 03 thành viên | **Môi trường:** PyCharm + Python 3.10+ + PyTorch + Hugging Face PEFT

---

## 1. Tổng quan dự án
Dự án nghiên cứu, hiện thực hóa từ gốc bằng PyTorch và đánh giá thực nghiệm các kỹ thuật **Parameter-Efficient Fine-Tuning (PEFT)** gồm **LoRA** (*Hu et al., ICLR 2022*) và **QLoRA** (*Dettmers et al., NeurIPS 2023*):
- Chứng minh toán học mức giảm tham số trên lớp tuyến tính $1024 \times 1024$ và phân rã VRAM mô hình 7B.
- Tự cài đặt lớp `LoRALinear` bằng PyTorch thuần (`src/lora_layer.py`) và kiểm chứng phép gộp trọng số suy luận $W_{\text{deploy}} = W_0 + \frac{\alpha}{r}BA$.
- Mô phỏng hội tụ trên bài toán thích ứng hạng nội tại thấp (`src/train_synthetic.py`).
- Mô phỏng lượng tử hóa 4-bit NormalFloat (NF4) và Double Quantization (`src/qlora_demo.py`).
- Huấn luyện đối chứng `distilbert-base-uncased` trên tập phân loại cảm xúc SST-2 (`src/train_hf_peft.py`).

---

## 2. Cấu trúc thư mục

```text
Project/
├── data/
│   └── sst2_sample.csv            # Tập dữ liệu mẫu SST-2 cân bằng 120 câu (60 pos / 60 neg)
├── figures/                       # Biểu đồ 300 DPI phục vụ Báo cáo & Slide
│   ├── lora_architecture.png      # Sơ đồ kiến trúc W_0 + (alpha/r) * BA & Merge
│   ├── params_comparison.png      # So sánh tham số lớp 1024x1024 (r=4, 8, 16)
│   ├── vram_breakdown.png         # Phân rã VRAM 7B: Full FT vs LoRA vs QLoRA
│   ├── loss_convergence.png       # Hội tụ hàm mất mát bài toán mô phỏng
│   └── hf_peft_convergence.png    # Kết quả huấn luyện thực tế DistilBERT PEFT
├── reports/
│   └── BaoCao_Nhom_S08.md         # Báo cáo nghiên cứu chuẩn IMRaD (6–10 trang A4)
├── results/                       # File JSON lưu log số liệu đo đạc thực nghiệm
├── slides/
│   └── Slide_Nhom_S08.md          # Kịch bản 10 slide thuyết trình & ngân hàng Q&A
├── src/
│   ├── __init__.py
│   ├── calculate_parameters.py    # Tính bảng tham số 1024x1024 & phân rã VRAM 7B
│   ├── lora_layer.py              # Cài đặt LoRALinear bằng PyTorch thuần + self-check
│   ├── plot_architecture.py       # Vẽ sơ đồ kiến trúc LoRA bằng matplotlib
│   ├── qlora_demo.py              # Mô phỏng NF4 + Double Quant & cấu hình HF QLoRA
│   ├── train_synthetic.py         # Thí nghiệm mô phỏng hạng nội tại r*=8
│   └── train_hf_peft.py           # Thí nghiệm thực tế DistilBERT (Full vs LoRA r=4, 16)
├── tests/
│   └── test_lora_layer.py         # Unit tests kiểm chứng LoRALinear & toán học tham số
├── overall.md                     # Tài liệu đặc tả tổng quan ban đầu
├── requirements.txt               # Danh sách thư viện Python
└── README.md                      # Tài liệu hướng dẫn này
```

---

## 3. Phân công công việc nhóm (03 thành viên)

| Thành viên | Vai trò | Nhiệm vụ chính |
| :--- | :--- | :--- |
| **Nguyễn Văn Minh Triều** *(Nhóm trưởng)* | Lead Kỹ thuật & Tổng hợp | Nghiên cứu *Hu et al. (2022)*, cài đặt `calculate_parameters.py`, phân tích đánh đổi VRAM/Optimizer States, chủ trì biên tập Báo cáo. |
| **Ngô Trọng Phúc** | Thực nghiệm & Trực quan hóa | Cài đặt `lora_layer.py`, `train_synthetic.py`, `train_hf_peft.py`, vẽ sơ đồ `W + BA`, đóng gói mã nguồn & viết kết quả thảo luận. |
| **Nguyễn Phúc Vương** | Nghiên cứu Mở rộng & Slide | Nghiên cứu *Dettmers et al. (2023)* (`qlora_demo.py`), so sánh 5 phương pháp, phân tích phản biện giới hạn LoRA, thiết kế bộ Slide. |

---

## 4. Hướng dẫn cài đặt & thực thi trên PyCharm (Windows)

### Bước 1: Tạo môi trường ảo và cài thư viện
Mở Terminal trong PyCharm (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
*(Nếu máy có GPU NVIDIA hỗ trợ CUDA 11.8, cài PyTorch bản CUDA trước: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)*

### Bước 2: Chạy Unit Tests kiểm chứng hệ thống
```powershell
python -m unittest discover -s tests -v
```

### Bước 3: Chạy các module thực nghiệm theo thứ tự
```powershell
# 1. Tính toán tham số 1024x1024, phân rã VRAM 7B & xuất biểu đồ
python src/calculate_parameters.py

# 2. Kiểm thử tự động LoRALinear (torch.allclose(y_1, y_2, atol=1e-5))
python src/lora_layer.py

# 3. Xuất sơ đồ kiến trúc LoRA
python src/plot_architecture.py

# 4. Thí nghiệm hội tụ bài toán mô phỏng hạng nội tại r*=8
python src/train_synthetic.py

# 5. Mô phỏng lượng tử hóa NF4 4-bit + Double Quantization
python src/qlora_demo.py

# 6. Huấn luyện DistilBERT trên dữ liệu SST-2 mẫu (chạy nhanh, không phụ thuộc tải dataset)
python src/train_hf_peft.py --quick --offline
```
Để chạy tập dữ liệu SST-2 đầy đủ hơn khi có kết nối mạng ổn định:
```powershell
python src/train_hf_peft.py --epochs 3 --max-train-samples 500 --max-eval-samples 200
```

---

## 5. Lưu ý kỹ thuật về QLoRA trên Windows
Thư viện `bitsandbytes` dùng nhân CUDA đặc thù tối ưu cho **Linux / Google Colab**. Khi chạy cục bộ trên Windows/CPU:
- Module `src/qlora_demo.py` tự động kích hoạt chế độ mô phỏng chuẩn xác thuật toán lượng tử hóa **NF4 (16 mức phân vị Gauss)** và **Double Quantization** bằng PyTorch thuần để đo lường sai số MSE/Cosine Similarity và tỷ lệ nén bit/tham số.
- Hàm `build_qlora_peft_config()` trong `src/qlora_demo.py` cung cấp sẵn cấu hình `BitsAndBytesConfig` (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`) để chạy trực tiếp nếu nhóm chuyển sang Google Colab có GPU T4/A100.
