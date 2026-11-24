# BỘ SLIDE THUYẾT TRÌNH SEMINAR (10 TRANG CHIẾU)
**ĐỀ TÀI S08: LoRA & QLoRA — TINH CHỈNH MÔ HÌNH LỚN HIỆU QUẢ VỀ SỐ LƯỢNG THAM SỐ**  
*Học phần: Trí tuệ Nhân tạo (UEH) | Thời lượng: 10 phút trình bày + 5 phút phản biện*

> **Phân công điều phối thuyết trình (mỗi thành viên ~3.3 phút):**
> - **Slide 1 – 4 (~3.3 phút):** Nguyễn Phúc Vương *(Động lực, So sánh 5 khái niệm & QLoRA)*
> - **Slide 5 – 7 (~3.3 phút):** Nguyễn Văn Minh Triều *(Cơ sở toán học LoRA, Ví dụ $1024 \times 1024$ & Bản chất VRAM)*
> - **Slide 8 – 10 (~3.4 phút):** Ngô Trọng Phúc *(Thực nghiệm PyTorch/PEFT, Kết quả đối chứng & Phản biện)*

---

## SLIDE 1: TRANG BÌA (TITLE SLIDE)
- **Tiêu đề lớn:** **LoRA & QLoRA: Tinh chỉnh Mô hình Lớn Hiệu quả về Số lượng Tham số**
- **Phụ đề:** Đề tài S08 — Học phần Trí tuệ Nhân tạo | Đại học Kinh tế TP. Hồ Chí Minh (UEH)
- **Nhóm thực hiện:**
  - Nguyễn Văn Minh Triều *(Nhóm trưởng — Toán học & Tổng hợp)*
  - Ngô Trọng Phúc *(Thực nghiệm & Trực quan hóa)*
  - Nguyễn Phúc Vương *(QLoRA, Phản biện & Slide)*

---

## SLIDE 2: NỖI ĐAU CỦA FULL FINE-TUNING TRÊN LLM (MOTIVATION)
*(Người trình bày: Nguyễn Phúc Vương)*
- **Mô hình 7B ở FP16** đòi hỏi tới **80–160 GB VRAM** để tinh chỉnh toàn phần:
  - Trọng số gốc: **14 GB**
  - Gradients: **14 GB**
  - AdamW Optimizer States (FP32 $m_t, v_t$): **56 GB**
  - Activations: nhiều GB theo context length
- **3 rào cản vận hành:**
  - Chi phí lưu trữ $N$ bản sao (14–100 GB/tác vụ) & độ trễ chuyển đổi serving.
  - **Catastrophic Forgetting:** Dễ mất tri thức tổng quát khi tinh chỉnh trên dữ liệu hẹp.
  - **Serial Adapters cũ:** Chèn lớp tuần tự $\rightarrow$ tăng độ trễ suy luận.

---

## SLIDE 3: SO SÁNH 5 PHƯƠNG PHÁP THÍCH ỨNG MÔ HÌNH
*(Người trình bày: Nguyễn Phúc Vương)*

| Phương pháp | Tham số cập nhật | VRAM huấn luyện | Lưu trữ đa tác vụ | Độ trễ suy luận |
| :--- | :---: | :---: | :---: | :---: |
| **Pre-training** | 100% (từ đầu) | Cực lớn | 1 bản gốc | Chuẩn |
| **Full Fine-tuning** | 100% | Rất lớn (80–160 GB) | $N$ bản sao lớn | Chuẩn (0 tăng thêm) |
| **Linear Probing** | Lớp đầu ra | Rất thấp | Rất gọn | Chuẩn |
| **Serial Adapters** | Lớp chèn tuần tự | Thấp | Gọn | **Tăng độ trễ** |
| **LoRA / QLoRA** | **Ma trận hạng thấp (< 1–3%)** | **Thấp / Rất thấp** | **Vài MB / tác vụ** | **Bằng 0 (sau gộp)** |

---

## SLIDE 4: ĐỘT PHÁ CỦA QLoRA (DETTMERS ET AL., 2023)
*(Người trình bày: Nguyễn Phúc Vương)*
- **Mục tiêu:** Giải quyết bài toán nén *chính trọng số gốc $W_0$* từ 16-bit xuống 4-bit mà không sụt giảm độ chính xác.
- **3 trụ cột kỹ thuật:**
  1. **4-bit NormalFloat (NF4):** Định dạng phân vị chuẩn tối ưu cho trọng số phân phối $\mathcal{N}(0, \sigma^2)$.
  2. **Double Quantization:** Lượng tử hóa luôn các hằng số absmax (tiết kiệm thêm **~0.373 bit/tham số**).
  3. **Paged Optimizers:** Tự động hoán đổi trang bộ nhớ GPU $\leftrightarrow$ CPU qua CUDA Unified Memory, chống lỗi OOM.

---

## SLIDE 5: CƠ CHẾ TOÁN HỌC CỦA LoRA (HU ET AL., 2022)
*(Người trình bày: Nguyễn Văn Minh Triều)*
- **Giả thuyết hạng nội tại:** Cập nhật trọng số $\Delta W$ có chiều nội tại rất nhỏ.
- **Công thức cốt lõi:**
  $$h = W_0 x + \frac{\alpha}{r} (B A) x$$
  - $W_0 \in \mathbb{R}^{d \times k}$: **Đóng băng** (`requires_grad = False`).
  - $A \in \mathbb{R}^{r \times k} \sim \mathcal{N}(0, \sigma^2)$ và $B \in \mathbb{R}^{d \times r} = 0$ (đảm bảo $\Delta W = 0$ lúc khởi đầu).
  - $r \ll \min(d, k)$ và hệ số tỷ lệ $\frac{\alpha}{r}$ giúp ổn định gradient khi đổi $r$.
- **Gộp trọng số suy luận:** $W_{\text{deploy}} = W_0 + \frac{\alpha}{r}BA \Rightarrow$ **Zero Inference Latency**.
- *Hình minh họa:* `figures/lora_architecture.png`

---

## SLIDE 6: VÍ DỤ TÍNH TOÁN SỐ ($1024 \times 1024$)
*(Người trình bày: Nguyễn Văn Minh Triều)*
- Xét 1 lớp tuyến tính $d = 1024, k = 1024$:
  - **Full Fine-tuning:** $1024 \times 1024 = \mathbf{1,048,576}$ tham số.
  - **LoRA ($r = 4$):** $4 \times (1024 + 1024) = \mathbf{8,192}$ tham số (**Giảm 99.22%**).
  - **LoRA ($r = 8$):** $8 \times (1024 + 1024) = \mathbf{16,384}$ tham số (**Giảm 98.44%**).
  - **LoRA ($r = 16$):** $16 \times (1024 + 1024) = \mathbf{32,768}$ tham số (**Giảm 96.88%**).
- *Biểu đồ cột:* `figures/params_comparison.png`

---

## SLIDE 7: PHẢN BIỆN MẤU CHỐT — THAM SỐ VS. VRAM THỰC TẾ
*(Người trình bày: Nguyễn Văn Minh Triều)*
- **Thông điệp cốt lõi:** **Giảm 99% tham số huấn luyện $\neq$ Giảm 99% VRAM GPU!**
- **Tại sao?**
  - Trọng số gốc $W_0$ (14 GB ở FP16) và Activations (~6 GB) **vẫn tồn tại trên VRAM**.
  - LoRA tiết kiệm chủ yếu nhờ **triệt tiêu >99% Optimizer States (56 GB) & Gradients (14 GB)** $\rightarrow$ Giảm ~77% tổng VRAM (90 GB $\rightarrow$ ~20.4 GB).
  - **QLoRA** nén tiếp $W_0$ xuống NF4 (~3.68 GB) $\rightarrow$ Giảm ~89% tổng VRAM (~10 GB), chạy được trên GPU 12–16 GB!
- *Biểu đồ phân rã VRAM:* `figures/vram_breakdown.png`

---

## SLIDE 8: KIỂM CHỨNG MÃ NGUỒN PYTORCH & MÔ PHỎNG
*(Người trình bày: Ngô Trọng Phúc)*
- **Cài đặt `LoRALinear` thuần PyTorch (`src/lora_layer.py`):**
  - Kiểm chứng phép gộp trọng số: `torch.allclose(y_1, y_2, atol=1e-5) == True`.
  - Unit tests xác nhận `base_layer.weight.grad is None`.
- **Thí nghiệm hạng nội tại thực $r^* = 8$ (`src/train_synthetic.py`):**
  - $r=4 < r^*$: hội tụ kém hơn do thiếu dung lượng (underfitting).
  - $r=8, 16$: hội tụ tiệm cận Full Fine-tuning với chi phí tham số chỉ ~6%–12%.
- *Biểu đồ hội tụ:* `figures/loss_convergence.png`

---

## SLIDE 9: THỰC NGHIỆM ĐỐI CHỨNG DISTILBERT TRÊN SST-2
*(Người trình bày: Ngô Trọng Phúc)*
- **Thiết lập:** Tinh chỉnh `distilbert-base-uncased` phân loại cảm xúc SST-2 (`q_lin`, `v_lin`).
- **So sánh 3 cấu hình:** Full Fine-tuning vs. LoRA ($r=4$) vs. LoRA ($r=16$).
- **Điểm nhấn kết quả:**
  - LoRA chỉ huấn luyện tỷ lệ rất nhỏ tham số so với Full Fine-tuning.
  - Độ chính xác (Accuracy / F1) bám sát Full Fine-tuning và hội tụ ổn định.
- *Biểu đồ thực nghiệm:* `figures/hf_peft_convergence.png` *(Cập nhật số liệu đo đạc thực tế khi trình chiếu).*

---

## SLIDE 10: KHI NÀO LoRA KHÔNG TỐI ƯU? & KẾT LUẬN
*(Người trình bày: Ngô Trọng Phúc)*
- **4 giới hạn cần lưu ý:**
  1. **Domain shift cực lớn:** Ngôn ngữ hiếm, mã hóa phi tự nhiên xa rời dữ liệu tiền huấn luyện.
  2. **Nạp tri thức sự kiện quy mô lớn:** Vượt quá dung lượng biểu diễn bậc thấp $r$.
  3. **Serving đa adapter không đồng nhất trong 1 batch:** Tốn chi phí tách nhánh tính toán nếu không gộp trước.
  4. **Ràng buộc phần cứng QLoRA:** Phụ thuộc CUDA kernels trên Linux (hạn chế trên Windows/CPU).
- **Kết luận:** LoRA/QLoRA là tiêu chuẩn vàng giúp dân chủ hóa việc tinh chỉnh LLM với chi phí thấp và không tăng độ trễ suy luận.

---

## PHỤ LỤC: NGÂN HÀNG CÂU HỎI PHẢN BIỆN SEMINAR (5 PHÚT HỎI ĐÁP)

1. **Tại sao ma trận $B$ phải khởi tạo bằng $0$ còn $A$ khởi tạo theo phân phối Gauss?**  
   *Gợi ý trả lời (Triều):* Để tại bước $0$, $\Delta W = BA = 0$, đảm bảo hàm đầu ra ban đầu giống hệt mô hình gốc $W_0$ mà không gây nhiễu ngẫu nhiên. Nếu cả $A$ và $B$ cùng bằng $0$, gradient truyền qua hai ma trận đều bằng $0$ và mô hình không thể học (symmetry trap).

2. **Ý nghĩa của hệ số tỷ lệ $\frac{\alpha}{r}$ là gì? Tại sao không giữ nguyên $\Delta W = BA$?**  
   *Gợi ý trả lời (Triều):* Khi tăng hoặc giảm hạng $r$, biên độ phương sai của tích $BA$ thay đổi tỷ lệ thuận theo $r$. Chia cho $r$ và nhân với hằng số $\alpha$ giúp chuẩn hóa độ lớn cập nhật, nhờ đó khi thử nghiệm nhiều mức $r$ khác nhau ta không phải tìm kiếm lại Learning Rate từ đầu.

3. **Tại sao giảm 99% tham số huấn luyện nhưng VRAM thực tế không giảm 99%?**  
   *Gợi ý trả lời (Triều):* Vì GPU vẫn bắt buộc phải giữ toàn bộ trọng số gốc $W_0$ (14 GB cho 7B FP16) và bộ nhớ kích hoạt trung gian (Activations) trong quá trình lan truyền thuận/ngược. Phần được cắt giảm mạnh là Gradients và AdamW Optimizer States (khoảng 75–78% tổng bộ nhớ Full Fine-tuning).

4. **Tại sao nói LoRA đạt "Zero Inference Latency" trong khi Serial Adapters lại làm chậm suy luận?**  
   *Gợi ý trả lời (Phúc):* Serial Adapters chèn thêm lớp phi tuyến tính nối tiếp nên không thể gộp đại số vào $W_0$. LoRA là phép cộng hai biến đổi tuyến tính cùng đầu vào $x$, do đó sau khi huấn luyện có thể cộng trực tiếp $W_{\text{deploy}} = W_0 + \frac{\alpha}{r}BA$ và vứt bỏ nhánh phụ khi suy luận.

5. **Trong code `LoRALinear`, tại sao phải tính `(x @ A^T) @ B^T` thay vì tính `B @ A` trước rồi mới nhân `x`?**  
   *Gợi ý trả lời (Phúc):* Nếu tính $B \cdot A$ trước trong lúc training, ta tạo ra ma trận trung gian kích thước $d \times k$ tốn bộ nhớ $O(dk)$ và độ phức tạp $O(rdk)$. Nhân tuần tự $x A^T$ (chiếu xuống chiều $r$) rồi nhân $B^T$ giảm bộ nhớ trung gian và phép tính xuống $O(r(d+k))$.

6. **Khác biệt cốt lõi giữa NF4 (NormalFloat4) và INT4 thông thường trong QLoRA là gì?**  
   *Gợi ý trả lời (Vương):* INT4 chia khoảng giá trị thành các bậc cách đều nhau, lãng phí độ chính xác ở hai đuôi nơi ít trọng số xuất hiện. NF4 chia 16 mức theo phân vị xác suất đều nhau của phân phối chuẩn $\mathcal{N}(0, 1)$, tối ưu về mặt lý thuyết thông tin cho trọng số mạng nơ-ron.

7. **Double Quantization tiết kiệm bộ nhớ bằng cách nào và giải quyết nhược điểm gì?**  
   *Gợi ý trả lời (Vương):* Khi lượng tử hóa theo từng khối 64 trọng số, mỗi khối tốn 1 hằng số FP32 (32 bit), tức mất thêm $0.5$ bit/tham số. Double Quantization nén chính các hằng số này xuống FP8 theo khối 256 hằng số, giảm phần overhead xuống còn $\sim 0.127$ bit/tham số.

8. **Nếu bài toán mới đòi hỏi dạy mô hình một ngôn ngữ lập trình hoàn toàn mới chưa từng có trong tiền huấn luyện, có nên dùng LoRA $r=4$ không?**  
   *Gợi ý trả lời (Vương):* Không nên. Đây là tình huống domain shift lớn và cần dung lượng biểu diễn cao; LoRA hạng thấp ($r=4$) sẽ bị underfitting nghiêm trọng. Khi đó cần dùng Full Fine-tuning (continual pre-training) hoặc tăng $r$ rất lớn kết hợp mở khóa lớp embedding/normalization.
