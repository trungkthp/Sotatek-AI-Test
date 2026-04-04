# 1. Sử dụng Python 3.12.3 để đồng bộ với môi trường máy tính của bạn
FROM python:3.12.3-slim

# 2. Cài đặt các thư viện hệ thống cần thiết cho OpenCV và build Detectron2
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Tải file model trực tiếp từ Google Drive (Giải quyết vấn đề file quá nặng trên GitHub)
# Thay 'YOUR_FILE_ID' bằng ID thực tế từ link chia sẻ Google Drive của bạn
RUN wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID' -O model_final.pth

# 4. Cài đặt Torch và torchvision bản CPU (để nhẹ và phù hợp gói Render Free)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 5. Cài đặt Detectron2 trực tiếp từ GitHub
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages

# 6. Cài đặt các thư viện còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# 7. Copy toàn bộ code vào container
COPY . .

# 8. Cấu hình cổng kết nối cho Streamlit
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
