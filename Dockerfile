# Sử dụng Python 3.10 mỏng nhẹ làm nền tảng
FROM python:3.10-slim

# 1. Cài đặt các thư viện hệ thống cần thiết cho OpenCV, Detectron2 và Build Tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# 2. Sao chép file requirements vào trước để tận dụng Docker cache
COPY requirements.txt .

# 3. Thực hiện các lệnh cài đặt "cực mạnh" theo yêu cầu của bạn
# Cài đặt các thư viện cơ bản
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Cài đặt Detectron2 trực tiếp từ GitHub
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages

# 4. Sao chép toàn bộ mã nguồn vào container
COPY . .

# Mở cổng 8501 (mặc định của Streamlit)
EXPOSE 8501

# Lệnh chạy ứng dụng khi container khởi động
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
