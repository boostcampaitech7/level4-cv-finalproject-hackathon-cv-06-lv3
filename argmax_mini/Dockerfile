# 단일 이미지 사용 (멀티스테이지 불필요)
FROM runpod/pytorch:3.10-2.0.1-117-devel

# GPG 키 갱신
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C \
    && apt-get update -o Acquire::AllowInsecureRepositories=true

# 시스템 패키지 설치
RUN apt-get install -y --no-install-recommends \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 환경 설정
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 실행 명령어
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]