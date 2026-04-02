# Sử dụng bản python slim để tiết kiệm dung lượng
FROM python:3.10-slim

# Cài đặt thư viện hệ thống tối thiểu cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đặt các thư viện theo thứ tự ưu tiên bản CPU nhẹ
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Ép cài Detectron2 bản build sẵn (chỉ mất vài giây)
RUN pip install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html

# Cài các thư viện còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render yêu cầu port mặc định thường là 10000 hoặc tự nhận diện
CMD ["streamlit", "run", "app.py", "--server.port", "10000", "--server.address", "0.0.0.0"]
