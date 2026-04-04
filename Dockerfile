# Sử dụng đúng bản 3.12.3 slim để nhẹ và đồng bộ với máy Trung
FROM python:3.12.3-slim

# Cài đặt các công cụ biên dịch (Cần thiết vì 3.12 phải build Detectron2 từ đầu)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# In ra version để kiểm tra chắc chắn trong Log Render
RUN python --version

# 1. Cài đặt Torch & torchvision (Bản cho CPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 2. Cài đặt Detectron2 từ GitHub (Sẽ mất khoảng 5-10 phút để biên dịch)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages

# 3. Cài đặt các thư viện từ requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# 4. Copy toàn bộ code vào
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
