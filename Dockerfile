# 1. Dùng bản Python có sẵn công cụ biên dịch
FROM python:3.10-slim

# 2. Cài đặt thư viện hệ thống cần thiết cho OpenCV và Detectron2
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Tạo thư mục làm việc
WORKDIR /app

# 4. Copy và cài đặt Requirements (Cài Torch trước)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Cài Detectron2 cưỡng chế
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 6. Copy toàn bộ code vào
COPY . .

# 7. Mở cổng cho Streamlit
EXPOSE 8501

# 8. Lệnh chạy app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
