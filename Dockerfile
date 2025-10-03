# 사용할 기본 이미지: Python 3.10.2 슬림 버전
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# 필요한 시스템 종속성 설치
# apt 캐시를 제거하여 이미지 크기를 최적화
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 파일을 복사하고 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 로컬의 checkpoints/wam_mit.pth 파일을 컨테이너의 /app/checkpoints/로 복사
COPY checkpoints/wam_mit.pth /app/checkpoints/

# 모든 소스 코드 복사
COPY . .

# CPU 개수 고정
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4

# 컨테이너 내부에서 Flask 앱이 사용할 포트 명시
EXPOSE 5000

# 컨테이너 시작 시 Flask 앱 실행
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "app:app"]