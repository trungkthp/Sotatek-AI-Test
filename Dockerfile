# 1. Sử dụng đúng bản Python 3.12.3-slim như máy local của bạn
FROM python:3.12.3-slim

# 2. Cài đặt các thư viện hệ thống cần thiết cho OpenCV, Git và biên dịch C++
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Cài đặt gdown để tải file từ Google Drive (Tránh lỗi Exit Code 8 của wget)
RUN pip install --no-cache-dir gdown --break-system-packages

# 4. Tải file model trực tiếp bằng ID (Thay YOUR_FILE_ID bằng ID thực tế của bạn)
# File sẽ được tải về và đặt tên là model_final.pth nằm ngay tại thư mục gốc /app
RUN gdown --id YOUR_FILE_ID -O model_final.pth

# 5. Cài đặt Torch và torchvision bản CPU (Bắt buộc cho Render gói Free để không tràn RAM)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 6. Cài đặt Detectron2 trực tiếp từ GitHub (Quá trình này có thể mất 5-10 phút)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages

# 7. Cài đặt các thư viện còn lại từ requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# 8. Copy toàn bộ mã nguồn vào Container
COPY . .

# 9. Mở cổng 8501 cho Streamlit
EXPOSE 8501

# 10. Lệnh chạy ứng dụng
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
