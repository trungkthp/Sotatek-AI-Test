FROM python:3.10-slim

# Cài đặt thư viện hệ thống (Thay thế cho packages.txt)
RUN apt-get update && apt-get install -y \
    build-essential libgl1 libglib2.0-0 git wget python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đặt Torch & Detectron2 bản build sẵn (Cực nhanh và chuẩn)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl

# Cài các thư viện còn lại (requirements.txt lúc này chỉ là phụ)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port mặc định của Render thường là 10000 hoặc bạn tự cấu hình
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
