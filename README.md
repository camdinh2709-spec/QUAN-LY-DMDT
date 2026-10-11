# 📈 HOSE V4 - QUANTITATIVE TOP N STOCKS SELECTION & PORTFOLIO OPTIMIZATION

Ứng dụng Web App xây dựng trên nền tảng **Streamlit** chuyên dụng cho việc kiểm định chiến lược định lượng (Quantitative Strategy Backtesting & Portfolio Optimization) trên thị trường chứng khoán Việt Nam (HOSE/HNX).

Ứng dụng giữ nguyên 100% logic thuật toán, mô hình toán học và quy trình kiểm định khoa học từ mô hình **HOSE V4 Generic Nested Walk-Forward**, tuân thủ nghiêm ngặt nguyên tắc **No Look-Ahead Bias** (không rò rỉ dữ liệu tương lai), khớp lệnh **Event-Driven tại Open(t+1)**, kết hợp quản trị rủi ro **Market Regime** và **Stock Active Mask**.

---

## 🌟 Điểm Nổi Bật Của Ứng Dụng

1. **Chuẩn hóa dữ liệu linh hoạt:** Tự động nhận diện các định dạng file OHLCV tiếng Việt có dấu/không dấu, hỗ trợ cả giá điều chỉnh (Adjusted) và giá khớp lệnh (Raw), tự động làm sạch và lọc dữ liệu lỗi.
2. **Hệ thống TA Score (0–10):** Chấm điểm kỹ thuật đa chiều bao gồm Trend Score (0–5), Momentum Score (0–3), Volume Score (0–1) và Volatility Score (0–1).
3. **Tối ưu hóa ngưỡng BUY/EXIT bằng Cross-Validation:** Tối ưu hóa duy nhất trên tập `Core Train` với cơ chế phạt độ biến thiên Sharpe Ratio (Stability Penalty), đảm bảo tính khái quát hóa cao.
4. **Cơ chế Nested Walk-Forward (Causal Design):** Tái lựa chọn TOP N cổ phiếu theo từng quý dựa trên dữ liệu đóng cửa phiên liền trước `AsOfDate (t-1)`, thích nghi theo chu kỳ thị trường nhưng không nhìn trước tương lai.
5. **So sánh 5 trường phái tối ưu hóa danh mục đầu tư:**
   - **Equal Weight (EW):** Phân bổ tỷ trọng đều.
   - **Mean-Variance Ledoit-Wolf (MPT-LW):** Tối ưu hóa Sharpe với ma trận hiệp phương sai co rút (Covariance Shrinkage) và L2 Regularization.
   - **Minimum Variance (MinVar-LW):** Tối thiểu hóa rủi ro danh mục.
   - **Hierarchical Risk Parity (HRP):** Phân bổ dựa trên phân cụm phân cấp tương quan.
   - **Blend 50/50 (Blend50):** Kết hợp cân bằng giữa MPT-LW và Equal Weight nhằm giảm sai số ước lượng.
6. **Quy trình khóa mô hình Champion khoa học:** Mô hình Champion được lựa chọn khách quan trên tập `Validation` thông qua điểm tổng hợp `ValidationScore` và **khóa cố định** trước khi đưa vào kiểm định ngoài mẫu `Final Test`.
7. **Quản trị rủi ro & Cash:** Tự động hạ tỷ trọng về Tiền mặt (Cash) khi thị trường vi phạm Market Regime (EMA200, Slope) hoặc cổ phiếu mất trạng thái kỹ thuật (Stock Active Mask).
8. **Xuất báo cáo đa chiều:** Xuất file Excel (`.xlsx`) đầy đủ 8 sheet tiêu chuẩn và các file `.csv` phục vụ báo cáo nghiên cứu.

---

## 📂 Cấu Trúc Mã Nguồn

Để triển khai lên GitHub và Streamlit Cloud, dự án gồm 3 file cốt lõi:

```text
QUAN LY DMDT/
├── app.py              # Toàn bộ mã nguồn ứng dụng Streamlit & thuật toán định lượng
├── requirements.txt    # Danh sách thư viện Python tương thích cho Streamlit Cloud
└── README.md           # Hướng dẫn chi tiết cài đặt, sử dụng và triển khai
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Python 3.10, 3.11 hoặc 3.12 (khuyến nghị Python 3.11).
- Trình quản lý gói `pip` hoặc `uv`.

### 2. Cài đặt các thư viện phụ thuộc
Mở terminal (PowerShell / Command Prompt / Bash) tại thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```

### 3. Khởi chạy Web App
Chạy lệnh sau trên terminal:

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở ứng dụng tại địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Tải Lên GitHub & Triển Khai Lên Streamlit Cloud

### Bước 1: Khởi tạo Git và đẩy mã nguồn lên GitHub
1. Mở Terminal tại thư mục chứa dự án:
```bash
git init
git add app.py requirements.txt README.md
git commit -m "Initial commit: Streamlit Quantitative App HOSE V4"
```
2. Tạo một Repository mới trên [GitHub](https://github.com/new) (ví dụ đặt tên: `hose-v4-quant-app`, chọn chế độ **Public**).
3. Liên kết và đẩy code lên GitHub:
```bash
git branch -M main
git remote add origin https://github.com/<tai-khoan-github-cua-ban>/hose-v4-quant-app.git
git push -u origin main
```

*(Lưu ý: Bạn có thể đưa thêm file dữ liệu mẫu `.csv` lên repo nếu muốn ứng dụng tự động nhận dữ liệu mẫu khi vừa mở lên).*

### Bước 2: Triển khai lên Streamlit Community Cloud (Miễn phí)
1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"Create app"** hoặc **"New app"**.
3. Chọn các thông số triển khai:
   - **Repository:** `<tai-khoan-github-cua-ban>/hose-v4-quant-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Nhấn **"Deploy!"**.
5. Trong vòng 1-2 phút, hệ thống Streamlit sẽ tự động cài đặt `requirements.txt` và khởi chạy Web App với đường dẫn công khai (ví dụ: `https://hose-v4-quant-app.streamlit.app`).

---

## 📊 Chuẩn Bị Dữ Liệu Đầu Vào

### 1. File dữ liệu OHLCV (Bắt buộc)
Hỗ trợ file định dạng `.csv`, `.xlsx`, `.xls` chứa các cột giá của cổ phiếu. Hệ thống tự động nhận diện các cột theo tên tiếng Anh hoặc tiếng Việt:

| Tên trường | Tên cột Tiếng Anh | Tên cột Tiếng Việt |
| :--- | :--- | :--- |
| **Mã cổ phiếu** | `Ticker`, `Symbol`, `Code` | `Mã`, `Mã CK` |
| **Ngày giao dịch** | `Date`, `Trading_Date`, `Time` | `Ngày`, `Thời gian` |
| **Giá mở cửa** | `Open`, `Open_Price` | `Giá mở cửa` |
| **Giá cao nhất** | `High`, `High_Price` | `Giá cao nhất` |
| **Giá thấp nhất** | `Low`, `Low_Price` | `Giá thấp nhất` |
| **Giá đóng cửa** | `Close`, `Close_Price` | `Giá đóng cửa` |
| **Giá điều chỉnh** | `adj_close`, `adjusted_close` | `Giá điều chỉnh` |
| **Khối lượng** | `Volume`, `Vol` | `Khối lượng`, `Khối lượng khớp lệnh` |

### 2. File Phân tích cơ bản - FA (Tùy chọn)
Chứa các chỉ tiêu tài chính point-in-time (`ROE`, `ROA`, `PE`, `PB`, `DebtToEquity`, `RevenueGrowth`,...) kèm cột ngày công bố BCTC (`Date` hoặc `Effective_Date`).

### 3. File Phân ngành - Sector (Tùy chọn)
Chứa ánh xạ giữa mã cổ phiếu và ngành nghề (`Ticker`, `Sector` hoặc `Ngành`) phục vụ kiểm soát hạn mức phân bổ theo ngành (Sector Cap).

---

## 📑 Hướng Dẫn Sử Dụng Các Tính Năng Trên Giao Diện

1. **Thanh bên trái (Sidebar):**
   - **Dữ liệu đầu vào:** Tải file OHLCV của bạn hoặc tích chọn *"Dùng file mẫu sẵn có"* nếu trong thư mục đã có file CSV.
   - **Phân chia thời gian:** Chọn chế độ `auto` (khuyến nghị - lấy năm cuối làm Final Test) hoặc `proportions` (chia theo % tỷ lệ).
   - **Tối ưu ngưỡng TA:** Thiết lập danh sách ngưỡng BUY/EXIT ứng viên và số fold Cross-Validation.
   - **Tiêu chí xếp hạng:** Tùy biến trọng số Trailing Sharpe, TA Score, Sức mạnh tương quan (RS), Low Volatility, Momentum, Liquidity.
   - **Quản trị rủi ro & Chi phí:** Thiết lập vốn ban đầu, phí giao dịch (0.15%), trượt giá (0.05%), bộ lọc Market Regime (EMA200).
   - Nhấn nút **"🚀 Chạy Kiểm Định Mô Hình"**.

2. **Các Tab kết quả:**
   - **Tab 1 - Final Test (OOS):** Thẻ KPI lợi nhuận, biểu đồ tăng trưởng NAV tương tác (Plotly), biểu đồ Drawdown, bảng so sánh mọi mô hình và nhật ký lệnh mua/bán chi tiết.
   - **Tab 2 - Validation & Champion:** Bảng chấm điểm xếp hạng mô hình và biểu đồ chọn lựa Champion.
   - **Tab 3 - Lịch trình TOP N:** Chi tiết danh sách Top N cổ phiếu được tái lựa chọn theo từng quý kèm điểm số chi tiết.
   - **Tab 4 - Tối ưu ngưỡng BUY/EXIT:** Kết quả chạy Cross-Validation trên Core Train để khóa ngưỡng tối ưu.
   - **Tab 5 - Phân tích kỹ thuật & TA:** Chọn mã cổ phiếu bất kỳ để xem nến giá, EMAs, MACD, RSI và điểm TA Score theo thời gian.
   - **Tab 6 - Tổng quan & Dữ liệu:** Bảng kiểm tra chất lượng dữ liệu, phiên giao dịch và thanh khoản từng mã.
   - **Tab 7 - Kiểm toán No-Leakage:** Bảng minh chứng chống rò rỉ thông tin tương lai.
   - **Tab 8 - Xuất báo cáo & File:** Tải file Excel đầy đủ 8 sheet và các file CSV phục vụ lưu trữ, báo cáo.

---

## 📐 Giải Thích Các Chỉ Số Đánh Giá

- **CAGR (Compound Annual Growth Rate):** Tỷ suất sinh lời kép hàng năm tính theo chu kỳ nắm giữ.
- **Sharpe Ratio:** Tỷ suất lợi nhuận trên độ biến động rủi ro tổng thể (Annualized).
- **Sortino Ratio:** Tỷ suất lợi nhuận trên độ biến động của các phiên giảm điểm (Downside Risk).
- **Maximum Drawdown (MaxDD):** Mức sụt giảm lớn nhất từ đỉnh giá trị tài sản ròng.
- **Calmar Ratio:** Tỷ lệ giữa CAGR và giá trị tuyệt đối của Max Drawdown.
- **Turnover (%):** Tỷ lệ vòng quay danh mục hàng năm tính theo tổng giá trị mua/bán chia cho vốn ban đầu.
- **Avg Exposure / Avg Cash (%):** Tỷ lệ trung bình nắm giữ cổ phiếu so với tiền mặt trong toàn bộ chu kỳ.
- **Validation Score:** Điểm số đa mục tiêu lựa chọn Champion:
  $$\text{Score} = 0.45 \times \text{Sharpe} + 0.15 \times \text{Sortino} + 0.012 \times (\text{Ret} - \text{Ret}_{\text{bench}}) + 0.004 \times \text{MaxDD} - 0.0005 \times \text{Turnover}$$
