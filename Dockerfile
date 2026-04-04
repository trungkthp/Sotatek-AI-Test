FROM python:3.10-slim

# 1. Cài đặt các thư viện hệ thống cần thiết cho OpenCV và Build Detectron2
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Nâng cấp pip và cài đặt Torch CPU trước (bản 2.1.0 để khớp với Detectron2 wheel)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Cài đặt Detectron2 bản build sẵn (Wheel) để tránh lỗi biên dịch và tiết kiệm RAM khi build
RUN pip install --no-cache-dir https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl

# 4. Sao chép và cài đặt các thư viện còn lại từ requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Sao chép toàn bộ mã nguồn vào image
COPY . .

# Mở cổng 8501 cho Streamlit
EXPOSE 8501

# Lệnh chạy ứng dụng
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
