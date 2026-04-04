# 1. Cài đặt các thư viện hệ thống cần thiết (Tương đương packages.txt)
RUN apt-get update && apt-get install -y \
    build-essential libgl1 libglib2.0-0 git wget python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Cài đặt Torch trước (Nền tảng cho Detectron2)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 3. LỆNH "SỨC MẠNH": Cài trực tiếp từ link Wheel của Meta
# Lệnh này bỏ qua việc kiểm tra môi trường hệ thống và cài bản build sẵn cực nhanh
RUN pip install --no-cache-dir \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl
