# Sử dụng Python 3.10 để đảm bảo tương thích tốt nhất
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bước 1: Cài đặt Torch và torchvision bản CPU trước
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Bước 2: ÉP CÀI Detectron2 trực tiếp từ GitHub bằng cách build tại chỗ
# Lệnh này sẽ tự động tải source và build để khớp hoàn toàn với môi trường Docker hiện tại
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Bước 3: Cài đặt các thư viện còn lại từ requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
