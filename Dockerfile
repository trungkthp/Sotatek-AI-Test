# Sử dụng Python 3.11 vì nó ổn định nhất cho Detectron2 hiện tại
FROM python:3.11-slim

# Tránh các câu hỏi tương tác khi cài apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Cài đặt các thư viện hệ thống (Cực kỳ quan trọng cho OpenCV và Detectron2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Nâng cấp pip và cài đặt các thư viện cơ bản trước
RUN pip install --no-cache-dir --upgrade pip

# 4. Copy file requirements và cài đặt (Tách riêng để tận dụng Docker Cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Lệnh "ép" cài Detectron2 từ GitHub (Do Docker có tài nguyên riêng nên không lo tràn RAM như Cloud)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 6. Copy toàn bộ mã nguồn vào Container
COPY . .

# 7. Mở cổng 8501 cho Streamlit
EXPOSE 8501

# 8. Lệnh khởi chạy app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
