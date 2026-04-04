# Đổi từ python-slim sang ubuntu cho ổn định
FROM ubuntu:22.04

# Tránh các câu hỏi tương tác khi cài đặt
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt Python và các thư viện hệ thống
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Cài đặt thư viện Python
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
