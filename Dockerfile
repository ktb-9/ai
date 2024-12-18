# Python 3.11 Slim 사용
FROM python:3.11-slim

# 컨테이너 작업 디렉토리 설정
WORKDIR /app

# 빌드에 필요한 패키지 및 OpenCV 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# requirements.txt만 먼저 복사하여 레이어 캐싱
COPY ./requirements.txt .

# 의존성 설치 (프리빌트 휠 우선 사용, GPU가 있으면 자동 감지)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# 나머지 애플리케이션 코드 복사
COPY . .

# FastAPI 서버 실행
EXPOSE 5001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
