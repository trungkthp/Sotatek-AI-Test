# Sử dụng Python 3.10 để đảm bảo tương thích tốt nhất với Detectron2 wheels
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và quá trình build
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bước 1: Cài đặt Torch và torchvision bản CPU
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Bước 2: Cài đặt Detectron2 từ kho wheel của Meta (Tránh lỗi 403 Forbidden)
RUN pip install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html

# Bước 3: Cài đặt các thư viện còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn và chạy ứng dụng
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
