# Sử dụng hình ảnh Python 3.14 mới nhất
FROM python:3.14-rc-slim

# Cài đặt các công cụ biên dịch cực kỳ quan trọng cho Python 3.14
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /code

# Cập nhật pip và các công cụ xây dựng gói
RUN python -m pip install --upgrade pip setuptools wheel

# Copy và cài đặt requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Chạy ứng dụng trên cổng của Hugging Face
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
