# 1. Sử dụng Python 3.12.3-slim đồng bộ với môi trường Ubuntu của bạn
FROM python:3.12.3-slim

# 2. Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Cài đặt gdown để tải file model từ Google Drive
RUN pip install --no-cache-dir --upgrade gdown --break-system-packages

# 4. TẢI MODEL: Đã điền ID từ link bạn cung cấp
RUN gdown --id 1voIzijFduwDECvD2OW_OUr4WONvWz9tW -O model_final.pth --confirm-inference

# 5. Cài đặt Torch bản CPU để phù hợp với giới hạn RAM của Render
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 6. Cài đặt Detectron2 từ GitHub
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages

# 7. Cài đặt các thư viện còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# 8. Copy toàn bộ code vào container
COPY . .

# 9. Cấu hình Port cho Streamlit
EXPOSE 8501

# 10. Chạy ứng dụng
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
