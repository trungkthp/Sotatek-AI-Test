FROM python:3.10-slim

# Cài thư viện hệ thống cho OpenCV
RUN apt-get update && apt-get install -y \
    build-essential libgl1 libglib2.0-0 git wget python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bước 1: Cài Torch CPU trước
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Bước 2: Cài Detectron2 từ link build sẵn (không phải biên dịch)
RUN pip install --no-cache-dir https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl

# Bước 3: Cài các thứ còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
