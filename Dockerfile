FROM python:3.12.3-slim

# 1. Cài đặt thư viện hệ thống (Đầy đủ nhất cho Build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Tạo user cho Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

# 3. Nâng cấp các công cụ Build (QUAN TRỌNG: Để tránh lỗi "getting requirements")
RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython --break-system-packages

# 4. Cài đặt Torch CPU bản mới nhất cho Python 3.12
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 5. Cài đặt Detectron2 (Sử dụng cờ --no-build-isolation để dùng chính setuptools đã cài ở bước 3)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages --no-build-isolation

# 6. Cài đặt các thư viện phụ trợ từ requirements.txt
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# 7. Copy code và model (Đảm bảo model_final.pth đã được upload lên Hugging Face)
COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
