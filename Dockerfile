FROM python:3.10-slim

# Cài thư viện hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài Torch và Detectron2 trực tiếp
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl

# Cài các thứ còn lại từ requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
