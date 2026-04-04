# Sử dụng Python 3.10-slim để khớp với bản Wheel ổn định nhất của Detectron2
FROM python:3.10-slim

# Chống lag và các câu hỏi xác nhận
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# ÉP đường dẫn thư viện để Python luôn thấy Detectron2
ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.10/site-packages"

# 1. Cài đặt thư viện hệ thống cực kỳ đầy đủ
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Nâng cấp pip và cài đặt Torch bản CPU trước
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. CÀI ĐẶT CƯỠNG CHẾ Detectron2 (Dùng link trực tiếp cho Python 3.10)
# Đây là mấu chốt để không bị lỗi No Module Found
RUN pip install --no-cache-dir https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl

# 4. Cài đặt các requirements còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code
COPY . .

EXPOSE 8501

# Sử dụng lệnh chạy tường minh để tránh nhầm Interpreter
CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
