# 사용할 기본 이미지: Python 3.10.2 슬림 버전
FROM pytorch/pytorch:2.7.0-cpu-py3.10

# 필요한 시스템 종속성 설치
# apt 캐시를 제거하여 이미지 크기를 최적화
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # OpenCV 의존성
    libgl1 \
    libglib2.0-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1 \
    libharfbuzz-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libsdl2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 파일을 복사하고 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 모든 소스 코드 복사
COPY . .

# 컨테이너 내부에서 Flask 앱이 사용할 포트 명시
EXPOSE 5000

# 컨테이너 시작 시 Flask 앱 실행
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "app:app"]