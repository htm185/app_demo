# V-Check (Adaptive News Verification)

Hệ thống **kiểm chứng tin tức đa tầng** (Vietnamese-first) sử dụng AI để phân loại nội dung thành **REAL / FAKE / NEI(UNCERTAIN)**, có thể kết hợp:

- **Layer 1**: mô hình lọc nhanh dựa trên *văn phong* + *chỉ số tương tác* (likes/comments/shares).
- **Layer 2**: *Retrieval* (Google CSE) + *NLI* (Natural Language Inference) để đối chiếu với các nguồn bên ngoài.

Repo gồm:
- **Backend API**: FastAPI (`main_api.py`)
- **Frontend UI**: Streamlit (`frontend_ui.py`)
- **Database**: SQLite + SQLAlchemy (`database.py`, file DB: `news1.db`)

---

## 1) Kiến trúc tổng quan

1. Người dùng nhập nội dung nghi ngờ + tham số tương tác trên UI Streamlit.
2. UI gọi API `POST /verify`.
3. API gọi `adaptive_predict()`:
   - Nếu Layer 1 tự tin (>= 0.90) → trả kết quả ngay.
   - Nếu chưa đủ tự tin → tìm bằng chứng (Google Search), cào nội dung trang, chạy NLI, tổng hợp xác suất.
4. Kết quả được **lưu vào SQLite** (bảng `news_history`).

---

## 2) Yêu cầu hệ thống

- Python 3.9+ (khuyến nghị)
- (Tùy chọn) CUDA nếu muốn tăng tốc PyTorch

Cài dependency:

```bash
pip install -r requirements.txt
```

> Lưu ý: repo hiện còn dùng thêm một số thư viện chưa có trong `requirements.txt` như `trafilatura`, `duckduckgo_search`, `pyvi`. Bạn nên bổ sung nếu chạy gặp lỗi import.

---

## 3) Cấu hình (Quan trọng)

### 3.1 Google Custom Search API
File: `search_module.py`

Bạn cần cấu hình:

- `GOOGLE_API_KEY`
- `GOOGLE_CSE_ID`

Nếu không cấu hình, Layer 2 sẽ không tìm được bằng chứng.

### 3.2 Model weights
File: `ai_models.py`

Mặc định code sẽ tìm các file:

- `model/best_model.pt` (Model 1)
- `model/best_model.pth` (Model 2)
- `model/meta_scaler.pkl` (chuẩn hoá metadata)

Nếu không có, hệ thống vẫn chạy nhưng sẽ dùng **weights ngẫu nhiên** (kết quả không đáng tin).

---

## 4) Chạy Backend (FastAPI)

```bash
uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

API endpoint:

- `POST /verify`

Ví dụ request:

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "content": "...",
    "likes": 1200,
    "comments": 350,
    "shares": 800
  }'
```

---

## 5) Chạy Frontend (Streamlit)

Mở một terminal khác:

```bash
streamlit run frontend_ui.py
```

UI mặc định gọi API tại:

- `http://127.0.0.1:8000/verify`

Nếu bạn đổi host/port backend, hãy sửa `API_URL` trong `frontend_ui.py`.

---

## 6) Database

- SQLite file: `news1.db`
- SQLAlchemy model: `NewsRecord` trong `database.py`
- Table: `news_history`

Có hàm reset DB (drop & create):

```python
from database import init_db
init_db()
```

> Cảnh báo: `init_db()` sẽ **xóa sạch dữ liệu cũ**.

---

## 7) Mô tả nhanh các file chính

- `main_api.py`: FastAPI app, định nghĩa endpoint `/verify`, lưu kết quả vào DB.
- `backend_logic.py`: pipeline chính `adaptive_predict()` + luật hậu xử lý (safety rules).
- `ai_models.py`: định nghĩa kiến trúc model và hàm `load_models()`.
- `search_module.py`: tìm bằng chứng qua Google CSE + chấm điểm nguồn.
- `evidence_processor.py`: cào nội dung trang + chọn đoạn liên quan + chạy NLI.
- `save_verdicts.py`: lưu verdict ra `data/verdicts.jsonl` và `data/verdicts.csv`.
- `frontend_ui.py`: UI Streamlit.

---

## 8) Troubleshooting

- **Không tìm được evidence**: kiểm tra `GOOGLE_API_KEY`/`GOOGLE_CSE_ID`, quota Google API, hoặc kết nối mạng.
- **Lỗi import** (`trafilatura`, `pyvi`, `duckduckgo_search`): cài thêm package tương ứng.
- **Kết quả vô nghĩa**: kiểm tra bạn đã đặt đúng file model weights trong thư mục `model/`.

---

## 9) Roadmap gợi ý (nếu bạn muốn cải thiện)

- Bổ sung `.env` và dùng `python-dotenv` để quản lý API key.
- Bổ sung `requirements.txt` đầy đủ.
- Tách cấu hình `CONFIDENCE_THRESHOLD`, `TEMPERATURE` sang config file.
- Thêm endpoint xem lịch sử kiểm chứng (`GET /history`).

---

## License

Chưa khai báo.
