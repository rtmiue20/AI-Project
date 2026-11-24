# TÀI LIỆU TỔNG QUAN DỰ ÁN (OVERALL PROJECT SPECIFICATION)
**ĐỀ TÀI S08: LoRA VÀ QLoRA - TINH CHỈNH MÔ HÌNH LỚN HIỆU QUẢ VỀ SỐ LƯỢNG THAM SỐ**
*Học phần: Trí tuệ Nhân tạo (Artificial Intelligence) - Đại học Kinh tế TP. Hồ Chí Minh (UEH)*
*Quy mô nhóm: 03 thành viên (Nhóm trưởng + 02 Thành viên)*
*Môi trường phát triển: PyCharm Professional / Community (Python 3.10+, PyTorch, Hugging Face PEFT)*

---

## 1. MÔ TẢ TỔNG QUAN DỰ ÁN & BỐI CẢNH (PROJECT OVERVIEW & CONTEXT)
Dự án tập trung nghiên cứu, hiện thực hóa và đánh giá thực nghiệm kỹ thuật **Tinh chỉnh hiệu quả về số lượng tham số (Parameter-Efficient Fine-Tuning - PEFT)**, với hai trọng tâm cốt lõi là **LoRA (Low-Rank Adaptation)** theo *Hu et al. (ICLR 2022)* và **QLoRA (Quantized Low-Rank Adaptation)** theo *Dettmers et al. (NeurIPS 2023)*.

* **Mục tiêu học thuật:** Giải thích cơ chế cập nhật ma trận trọng số hạng thấp, định lượng mức độ giảm tải tham số huấn luyện, phân biệt ranh giới giữa LoRA, QLoRA và Full Fine-tuning, đồng thời chứng minh bằng tính toán số học và thực nghiệm trên mã nguồn.
* **Sản phẩm bàn giao chuẩn UEH:**
  1. **Báo cáo nghiên cứu khoa học:** 6–10 trang A4 tiếng Việt (chuẩn IMRaD: Tóm tắt, Giới thiệu, Cơ sở lý thuyết, Phương pháp/Ví dụ tính toán, Kết quả & Thảo luận, Kết luận, Tài liệu tham khảo).
  2. **Slide thuyết trình Seminar:** 8–12 trang chiếu, chuẩn bị cho 10 phút trình bày và 5 phút phản biện hỏi đáp.
  3. **Mã nguồn thực nghiệm PyCharm:** Dự án Python hoàn chỉnh gồm module tính toán số ma trận, module cài đặt LoRA thuần và kịch bản huấn luyện PEFT có kiểm chứng.

---

## 2. NỖI ĐAU THỰC TẾ & ĐỘNG LỰC NGHIÊN CỨU (PAIN POINTS & MOTIVATION)

### 2.1. Nỗi đau hiện tại của bài toán Tinh chỉnh Mô hình Lớn (LLMs)
* **Khủng hoảng bộ nhớ phần cứng (VRAM Bottleneck):** Khi tinh chỉnh toàn bộ mô hình (Full Fine-tuning), ngoài việc lưu trữ trọng số mô hình (ví dụ mô hình 7B ở fp16 tốn ~14GB VRAM), hệ thống bắt buộc phải lưu trữ:
  * Gradients (đạo hàm): ~14GB VRAM.
  * Trạng thái bộ tối ưu (Optimizer States - AdamW lưu $m_t$ và $v_t$ ở fp32): tiêu tốn $2 \times 4 \text{ bytes} \times 7\text{B} = 56\text{GB}$ VRAM.
  * Bộ nhớ trung gian kích hoạt (Activations): hàng chục GB tùy theo context length.
  $\rightarrow$ Tinh chỉnh một mô hình 7B tham số đòi hỏi cụm máy chủ 80GB–160GB VRAM (A100/H100), hoàn toàn bất khả thi đối với sinh viên, nhóm nghiên cứu nhỏ và doanh nghiệp vừa/nhỏ.
* **Chi phí lưu trữ & triển khai đa tác vụ (Storage & Serving Redundancy):** Nếu áp dụng Full Fine-tuning cho $N$ bài toán khác nhau (phân loại cảm xúc, tóm tắt y tế, chatbot tài chính), ta phải lưu trữ $N$ bản sao mô hình độc lập (mỗi bản nặng 14GB–100GB). Khi phục vụ suy luận (serving), việc nạp/chuyển đổi qua lại giữa các mô hình lớn gây ra độ trễ chuyển đổi cực cao.
* **Hiện tượng quên thảm khốc (Catastrophic Forgetting):** Cập nhật toàn bộ trọng số trên tập dữ liệu hẹp dễ làm mất đi tri thức tổng quát quý giá đã học được trong giai đoạn Pre-training.
* **Hạn chế của các phương pháp Adapter cổ điển:** Các kiến trúc Adapter trước đây chèn thêm các lớp tuần tự (serial layers) vào giữa các khối Transformer, dẫn đến tăng độ trễ suy luận (inference latency) đáng kể do không thể song song hóa.

---

## 3. HƯỚNG ĐI & CÁCH TIẾP CẬN KỸ THUẬT (APPROACH & METHODOLOGY)

### 3.1. Cơ sở toán học của LoRA (Low-Rank Adaptation)
* **Giả thuyết hạng nội tại (Intrinsic Rank Hypothesis):** Sự thay đổi trọng số trong quá trình thích ứng tác vụ $\Delta W$ có "chiều nội tại" rất thấp so với không gian tham số ban đầu.
* **Cơ chế phân rã ma trận:**
  Thay vì cập nhật trực tiếp ma trận trọng số gốc $W_0 \in \mathbb{R}^{d \times k}$, ta đóng băng hoàn toàn $W_0$ và phân rã phần biến thiên trọng số $\Delta W$ thành tích của hai ma trận hạng thấp:
  $$\Delta W = B \cdot A$$
  Trong đó:
  * $W_0 \in \mathbb{R}^{d \times k}$ (Đóng băng, `requires_grad = False`).
  * $A \in \mathbb{R}^{r \times k}$ (Khởi tạo theo phân phối Gauss $\mathcal{N}(0, \sigma^2)$).
  * $B \in \mathbb{R}^{d \times r}$ (Khởi tạo ban đầu bằng 0, đảm bảo $\Delta W = 0$ tại thời điểm bắt đầu huấn luyện).
  * Hạng $r \ll \min(d, k)$ (thường chọn $r \in \{4, 8, 16\}$).
* **Phương trình đầu ra và hệ số tỷ lệ $\alpha$ (Scaling Factor):**
  $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$
  $\alpha$ là siêu tham số không đổi giúp cố định độ lớn của bản cập nhật khi thay đổi hạng $r$, tránh việc phải chỉnh lại Learning Rate khi tinh chỉnh $r$.
* **Tính năng gộp trọng số (Zero Inference Latency):** Khi triển khai suy luận, ta có thể cộng trực tiếp trọng số:
  $$W_{\text{deploy}} = W_0 + \frac{\alpha}{r} B A$$
  Do đó mô hình suy luận hoàn toàn không tốn thêm bộ nhớ hay độ trễ so với mô hình gốc.

### 3.2. Cơ chế lượng tử hóa QLoRA (Quantized LoRA)
* **Lượng tử hóa 4-bit NormalFloat (NF4):** Nén ma trận trọng số gốc $W_0$ xuống định dạng 4-bit tối ưu thông tin cho phân phối chuẩn.
* **Double Quantization (Lượng tử hóa kép):** Lượng tử hóa cả các hằng số lượng tử (quantization constants), tiết kiệm thêm ~0.37 bit/tham số.
* **Paged Optimizers:** Quản lý cấp phát trang bộ nhớ giữa CPU và GPU qua CUDA Unified Memory để xử lý các đỉnh bộ nhớ (memory spikes), chống lỗi CUDA Out of Memory (OOM).

---

## 4. VÍ DỤ TÍNH TOÁN SỐ BẮT BUỘC (MATHEMATICAL VERIFICATION)
*Xét một lớp tuyến tính Transformer kích thước $1024 \times 1024$ ($d = 1024, k = 1024$):*

1. **Full Fine-tuning:**
   $$\text{Params}_{\text{Full}} = d \times k = 1024 \times 1024 = 1,048,576 \text{ tham số}$$
2. **LoRA với $r = 4$:**
   $$\text{Params}_{\text{LoRA(r=4)}} = r \times (d + k) = 4 \times (1024 + 1024) = 8,192 \text{ tham số}$$
   $$\rightarrow \text{Tỷ lệ giảm: } \frac{1,048,576 - 8,192}{1,048,576} \approx 99.22\%$$
3. **LoRA với $r = 8$:**
   $$\text{Params}_{\text{LoRA(r=8)}} = 8 \times (1024 + 1024) = 16,384 \text{ tham số}$$
   $$\rightarrow \text{Tỷ lệ giảm: } \frac{1,048,576 - 16,384}{1,048,576} \approx 98.44\%$$
4. **LoRA với $r = 16$:**
   $$\text{Params}_{\text{LoRA(r=16)}} = 16 \times (1024 + 1024) = 32,768 \text{ tham số}$$
   $$\rightarrow \text{Tỷ lệ giảm: } \frac{1,048,576 - 32,768}{1,048,576} \approx 96.88\%$$

* **Thảo luận phản biện mấu chốt (Rubric UEH):** Giảm 99% tham số huấn luyện **không đồng nghĩa** với việc giảm 99% VRAM GPU thực tế. VRAM vẫn phải gánh trọng số mô hình gốc (base model weights) và kích hoạt trung gian (activations). Mức tiết kiệm VRAM chủ yếu đến từ việc triệt tiêu ~75% bộ nhớ của Optimizer States và Gradients.

---

## 5. RỦI RO GẶP PHẢI & CÁCH XỬ LÝ (RISKS & MITIGATION STRATEGIES)

| STT | Rủi ro kỹ thuật / Thực nghiệm | Hậu quả | Giải pháp xử lý cụ thể |
| :--- | :--- | :--- | :--- |
| **1** | **Tràn bộ nhớ GPU (CUDA OOM)** khi chạy mô hình lớn cục bộ. | Code crash, treo máy tính cá nhân. | Sử dụng mô hình nhẹ (`distilbert-base-uncased` hoặc `gpt2`), giảm `per_device_train_batch_size = 8`, bật `gradient_accumulation_steps = 4`. Nếu không có GPU rời, chạy chế độ mô phỏng `LoRALinear` trên CPU. |
| **2** | **Lỗi không tương thích thư viện `bitsandbytes`** trên hệ điều hành Windows khi làm QLoRA. | Không nạp được `bitsandbytes` hoặc lỗi CUDA DLLs trên Windows. | Ưu tiên dùng bản `bitsandbytes-windows` hoặc chuyển module thực nghiệm QLoRA sang Google Colab / Linux container, PyCharm giữ vai trò code và phân tích dữ liệu. |
| **3** | **Quá khớp (Overfitting) hoặc Suy giảm hiệu năng** khi chọn $r$ quá nhỏ hoặc quá lớn. | $r=4$ mô hình học kém (underfitting), $r=64$ tốn tài nguyên và dễ overfit. | Khảo sát thực nghiệm với dải $r \in \{4, 8, 16\}$, cố định $\alpha = 2r$ hoặc $\alpha = 16$, theo dõi sát sao validation loss. |
| **4** | **Báo cáo vượt quá số trang quy định** (vi phạm quy cách 6–10 trang A4). | Bị trừ điểm hình thức đồ án. | Tách bạch phần phụ lục (code chi tiết, log kết quả) ra ngoài; phần chính chỉ giữ bảng tổng hợp, biểu đồ trực quan và công thức cốt lõi. |

---

## 6. HƯỚNG TRIỂN KHAI CHI TIẾT TRÊN PYCHARM (PYCHARM WORKFLOW)

### 6.1. Cấu trúc thư mục chuẩn (Project Directory Structure)
```text
LoRA_QLoRA_UEH/
├── data/                        # Tập dữ liệu mẫu (SST-2 hoặc IMDb sample)
├── figures/                     # Chứa biểu đồ xuất ra (.png) chèn vào Báo cáo & Slide
│   ├── params_comparison.png
│   ├── loss_convergence.png
│   └── lora_architecture.png
├── src/
│   ├── __init__.py
│   ├── lora_layer.py            # Cài đặt thủ công lớp LoRALinear bằng PyTorch thuần
│   ├── calculate_parameters.py  # Script tự động tính bảng tham số 1024x1024
│   ├── train_synthetic.py       # Huấn luyện mô phỏng lớp tuyến tính kiểm chứng hội tụ
│   └── train_hf_peft.py         # Huấn luyện thực tế mô hình DistilBERT với PEFT LoRA
├── reports/
│   └── BaoCao_NhomX_S08.pdf     # File báo cáo hoàn chỉnh (Overleaf / Word xuất ra)
├── slides/
│   └── Slide_NhomX_S08.pdf      # Slide thuyết trình (8-12 trang)
├── requirements.txt             # Danh sách dependencies
└── README.md                    # Hướng dẫn chạy và giải thích kỹ thuật
```

### 6.2. Các bước thiết lập & thực thi trên PyCharm
1. **Thiết lập môi trường ảo:**
   * Mở PyCharm $\rightarrow$ `New Project` $\rightarrow$ Chọn `Virtualenv` (Python 3.10 hoặc 3.11).
2. **Cài đặt dependencies qua Terminal của PyCharm:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # nếu có CUDA
   pip install transformers datasets peft accelerate evaluate scikit-learn matplotlib seaborn tabulate
   ```
3. **Thực thi Module 1 (`calculate_parameters.py`):**
   * Chạy tính toán ma trận $1024 \times 1024$ cho các rank $r \in \{4, 8, 16\}$, in bảng Markdown/LaTeX và vẽ đồ thị bar chart `params_comparison.png`.
4. **Thực thi Module 2 (`lora_layer.py`):**
   * Kiểm thử tính đúng đắn của phép gộp trọng số:
     $$y_1 = x W_0^T + \frac{\alpha}{r} x (B A)^T$$
     $$y_2 = x (W_0 + \frac{\alpha}{r} B A)^T$$
     Khẳng định `torch.allclose(y_1, y_2, atol=1e-5)` trả về `True`.
5. **Thực thi Module 3 (`train_hf_peft.py`):**
   * Huấn luyện mô hình phân loại văn bản nhỏ với 3 cấu hình: Full Fine-tuning vs LoRA $r=4$ vs LoRA $r=16$.
   * Ghi nhận log: thời gian huấn luyện, đỉnh VRAM, Accuracy, xuất đồ thị hội tụ vào thư mục `figures/`.

---

## 7. PHÂN CHIA CÔNG VIỆC CHO 3 THÀNH VIÊN (TEAM WORKLOAD MATRIX)

| Thành viên                              | Vai trò | Nhiệm vụ kỹ thuật (Code & Lý thuyết) | Nhiệm vụ Báo cáo & Slide | Phản biện Seminar |
|:----------------------------------------| :--- | :--- | :--- | :--- |
| **Nguyễn Văn Minh Triều - Nhóm trưởng** | **Lead Kỹ thuật & Tổng hợp dự án** | - Đọc sâu bài báo gốc *Hu et al. (2022)*.<br>- Viết script `calculate_parameters.py` tính toán ma trận $1024 \times 1024$.<br>- Phân tích sự đánh đổi VRAM/GPU và Optimizer states. | - Chủ trì biên tập Báo cáo (6–10 trang).<br>- Viết phần: Tóm tắt, Phương pháp & Ví dụ tính toán số.<br>- Rà soát chuẩn rubric UEH. | Trình bày phần Cơ sở Toán học & Trả lời các câu hỏi về công thức ma trận, rank $r$. |
| **Ngô Trọng Phúc**                      | **Thực nghiệm & Trực quan hóa** | - Cài đặt module `lora_layer.py` (PyTorch thuần).<br>- Chạy script thực nghiệm đối chứng `train_hf_peft.py`.<br>- Vẽ sơ đồ kiến trúc $W + BA$ và xuất các biểu đồ kết quả. | - Viết phần: Kết quả và Thảo luận.<br>- Đóng gói mã nguồn, làm sạch chú thích code và tạo file README chi tiết. | Trình bày phần Thực nghiệm & Trả lời câu hỏi về gộp trọng số, huấn luyện, đánh đổi bộ nhớ. |
| **Nguyễn Phúc Vương**                   | **Nghiên cứu Mở rộng & Thiết kế Slide** | - Đọc bài báo *Dettmers et al. (2023)* về QLoRA.<br>- Phân tích so sánh 5 khái niệm: Pre-training, Full Fine-tuning, Linear probing, Adapters, PEFT.<br>- Tổng hợp các hạn chế khi LoRA không tối ưu. | - Thiết kế bộ Slide thuyết trình (8–12 slide).<br>- Viết phần: Giới thiệu, Cơ sở lý thuyết, Phân tích phản biện & Kết luận. | Trình bày phần Động lực, QLoRA, Phản biện & Trả lời câu hỏi về điều kiện áp dụng LoRA/QLoRA. |

---

## 8. KẾT QUẢ MONG ĐỢI & CHECKLIST ĐẠT ĐIỂM TỐI ĐA (EXPECTED DELIVERABLES & SCORING CRITERIA)

* [ ] **Báo cáo chuẩn học thuật 6–10 trang:**
  * Có đầy đủ công thức toán học $W' = W + \frac{\alpha}{r}BA$ với định nghĩa từng biến.
  * Có bảng số liệu tham số $1024 \times 1024$ chính xác và thảo luận bản chất VRAM.
  * Có sơ đồ kiến trúc tự vẽ rõ nét, trích dẫn chuẩn IEEE/APA.
  * Có phân tích phản biện: LoRA không hiệu quả khi nào (khi bài toán mới khác biệt hoàn toàn miền dữ liệu gốc, hoặc khi cần bổ sung kiến thức sự kiện hoàn toàn mới).
* [ ] **Slide thuyết trình 8–12 trang:**
  * Bố cục chuẩn mực, không quá tải chữ, làm nổi bật biểu đồ so sánh.
  * Phân chia thời lượng nói đồng đều cho cả 3 thành viên (mỗi bạn ~3.3 phút).
* [ ] **Phần hỏi đáp phản biện (20 điểm):**
  * Cả 3 thành viên đều chủ động nắm chắc phần của nhau, mỗi người trả lời dứt khoát ít nhất 1 câu hỏi từ hội đồng.
