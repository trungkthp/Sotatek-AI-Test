# 1. Dùng Python 3.11 bản Slim cho nhẹ nhưng vẫn đủ công cụ
FROM python:3.11-slim

# 2. Thiết lập biến môi trường để cài đặt không bị hỏi (Non-interactive)
ENV DEBIAN_FRONTEND=noninteractive

# 3. Cài đặt các thư viện hệ thống CẦN THIẾT (Không được thiếu cái nào)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Nâng cấp pip và cài đặt Torch/Torchvision TRƯỚC (Bắt buộc có trước Detectron2)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. ÉP CÀI Detectron2 bằng link trực tiếp (Bản CPU dành cho Linux Python 3.11)
# Cách này bỏ qua việc tìm kiếm trên PyPI, tải thẳng file chuẩn về cài
RUN pip install --no-cache-dir https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp311-cp311-linux_x86_64.whl

# 6. Cài đặt các requirements còn lại (Bỏ detectron2 khỏi file txt nhé)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy code và chạy app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
